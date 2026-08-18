//! Reading OpenRAO's own JSON CRAC format.
//!
//! This is what turns 428 CRAC files in the vendored reference checkout into
//! usable input, and with them the expected margins, costs and activated
//! actions their scenarios state.
//!
//! # The format has 24 versions and they renamed things
//!
//! Across the corpus the same concept appears under several spellings, because
//! the format evolved:
//!
//! | Concept | 1.x spelling | 2.x spelling |
//! |---|---|---|
//! | available at an instant | `freeToUseUsageRules` | `onInstantUsageRules` |
//! | available in a state | `onStateUsageRules` | `onContingencyStateUsageRules` |
//! | conditional on a CNEC | `onFlowConstraintUsageRules` | `onConstraintUsageRules` |
//! | connect/disconnect | `topologicalActions` | `terminalsConnectionActions` |
//! | which end of a branch | `1` / `2` | `"left"` / `"right"` |
//!
//! and older files omit `instants` entirely, having had a fixed four. So this
//! reader accepts every spelling it has seen and **reports what it could not
//! place** rather than failing, on the same principle as
//! [`crate::iidm`]: a reader pinned to one version handles a third of the
//! material.
//!
//! # What is deliberately lossy
//!
//! Angle and voltage CNECs are counted, not converted — see
//! [`Crac::angle_cnecs`]. Counter-trade range actions are carried but the
//! optimizer does not model them. Both are stated in the model rather than
//! hidden here.

use std::collections::HashMap;

use serde_json::Value;

use super::crac::*;

#[derive(Debug)]
pub enum CracError {
    Json(serde_json::Error),
    /// The document is not a CRAC at all.
    NotACrac,
    /// A CNEC or usage rule named an instant the document never declared.
    UnknownInstant { referrer: String, instant: String },
    /// A CNEC or usage rule named a contingency the document never declared.
    UnknownContingency { referrer: String, contingency: String },
}

impl std::fmt::Display for CracError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CracError::Json(e) => write!(f, "malformed CRAC JSON: {e}"),
            CracError::NotACrac => write!(f, "document is not a CRAC"),
            CracError::UnknownInstant { referrer, instant } => {
                write!(f, "`{referrer}` refers to undeclared instant `{instant}`")
            }
            CracError::UnknownContingency { referrer, contingency } => {
                write!(f, "`{referrer}` refers to undeclared contingency `{contingency}`")
            }
        }
    }
}

impl std::error::Error for CracError {}

impl From<serde_json::Error> for CracError {
    fn from(e: serde_json::Error) -> Self {
        CracError::Json(e)
    }
}

/// What the reader could not place.
///
/// Returned rather than logged, for the reason the whole importer layer keeps
/// returning these: a remedial action silently dropped from a CRAC makes the
/// optimizer report that no action helps, which looks like an answer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CracReport {
    /// The `version` field the document declared.
    pub version: Option<String>,
    /// Elementary-action spellings encountered but not understood, with counts.
    pub unknown_actions: HashMap<String, usize>,
    /// Usage-rule spellings encountered but not understood, with counts.
    pub unknown_usage_rules: HashMap<String, usize>,
    /// Remedial actions dropped because none of their usage rules could be
    /// resolved, or because they had no elementary action left.
    pub dropped_actions: Vec<String>,
    /// Angle and voltage CNECs, counted rather than converted.
    pub angle_cnecs: usize,
    pub voltage_cnecs: usize,
}

impl CracReport {
    /// True when nothing was skipped — the state a caller should normally
    /// insist on before trusting an optimization built on this CRAC.
    pub fn is_complete(&self) -> bool {
        self.unknown_actions.is_empty()
            && self.unknown_usage_rules.is_empty()
            && self.dropped_actions.is_empty()
    }
}

/// Read an OpenRAO JSON CRAC.
pub fn parse(text: &str) -> Result<(Crac, CracReport), CracError> {
    let doc: Value = match serde_json::from_str(text) {
        Ok(doc) => doc,
        // 98 of the 428 CRACs in the reference checkout are not valid JSON:
        // they contain bare `NaN` (and occasionally `Infinity`), which Java's
        // Jackson writes by default and no JSON parser accepts. Retrying with
        // those literals neutralised is the difference between reading a third
        // of the corpus and reading all of it.
        Err(_) => serde_json::from_str(&neutralise_non_finite(text))?,
    };
    from_value(&doc)
}

/// Replace bare `NaN`, `Infinity` and `-Infinity` tokens with `null`.
///
/// Both mean "no value stated" everywhere they appear in a CRAC — an absent
/// rated current, an unbounded range — which is exactly what `null` means to
/// the readers below, since they filter non-finite numbers out anyway.
///
/// String contents are left alone: a network element legitimately named
/// `"NaN Substation"` must not be rewritten. That is the whole reason this is a
/// scanner rather than a search-and-replace.
fn neutralise_non_finite(text: &str) -> String {
    const TOKENS: [&str; 3] = ["-Infinity", "Infinity", "NaN"];
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut in_string = false;

    while let Some(c) = rest.chars().next() {
        if in_string {
            // Copy an escape pair as a unit so a `\"` does not look like the
            // end of the string.
            if c == '\\' {
                let mut it = rest.chars();
                out.push(it.next().unwrap());
                if let Some(escaped) = it.next() {
                    out.push(escaped);
                }
                rest = it.as_str();
                continue;
            }
            if c == '"' {
                in_string = false;
            }
            out.push(c);
            rest = &rest[c.len_utf8()..];
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            rest = &rest[1..];
            continue;
        }
        // Only replace a whole token, so an identifier merely starting with one
        // is left alone.
        let token = TOKENS.iter().find(|t| {
            rest.strip_prefix(**t).is_some_and(|after| {
                after.chars().next().is_none_or(|n| !n.is_alphanumeric() && n != '_')
            })
        });
        match token {
            Some(t) => {
                out.push_str("null");
                rest = &rest[t.len()..];
            }
            None => {
                out.push(c);
                rest = &rest[c.len_utf8()..];
            }
        }
    }
    out
}

pub fn read(path: impl AsRef<std::path::Path>) -> Result<(Crac, CracReport), CracError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        CracError::Json(serde_json::Error::io(e))
    })?;
    parse(&text)
}

/// Whether a JSON document looks like a CRAC, without committing to reading it.
pub fn is_crac(doc: &Value) -> bool {
    doc.get("type").and_then(Value::as_str) == Some("CRAC")
}

fn text(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_string)
}

fn number(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
}

fn integer(v: &Value, key: &str) -> Option<i64> {
    v.get(key).and_then(Value::as_i64)
}

fn array<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

/// The four instants every pre-2.0 CRAC had implicitly.
fn default_instants() -> Vec<Instant> {
    vec![
        Instant { id: "preventive".into(), kind: InstantKind::Preventive },
        Instant { id: "outage".into(), kind: InstantKind::Outage },
        Instant { id: "auto".into(), kind: InstantKind::Auto },
        Instant { id: "curative".into(), kind: InstantKind::Curative },
    ]
}

pub fn from_value(doc: &Value) -> Result<(Crac, CracReport), CracError> {
    if !is_crac(doc) {
        return Err(CracError::NotACrac);
    }
    let mut report = CracReport { version: text(doc, "version"), ..Default::default() };

    // Instants. Older versions omit the list and rely on four fixed ones; the
    // rest of the reader resolves by name either way, so supplying the defaults
    // here is all the compatibility that costs.
    let declared = array(doc, "instants");
    let instants: Vec<Instant> = if declared.is_empty() {
        default_instants()
    } else {
        declared
            .iter()
            .filter_map(|i| {
                Some(Instant {
                    id: text(i, "id")?,
                    kind: match text(i, "kind")?.to_ascii_uppercase().as_str() {
                        "PREVENTIVE" => InstantKind::Preventive,
                        "OUTAGE" => InstantKind::Outage,
                        "AUTO" => InstantKind::Auto,
                        _ => InstantKind::Curative,
                    },
                })
            })
            .collect()
    };

    let contingencies: Vec<Contingency> = array(doc, "contingencies")
        .iter()
        .filter_map(|c| {
            Some(Contingency {
                id: text(c, "id")?,
                name: text(c, "name"),
                elements: array(c, "networkElementsIds")
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect(),
            })
        })
        .collect();

    let index = Index { instants: &instants, contingencies: &contingencies };

    let mut flow_cnecs = Vec::new();
    for c in array(doc, "flowCnecs") {
        if let Some(cnec) = read_flow_cnec(c, &index)? {
            flow_cnecs.push(cnec);
        }
    }
    report.angle_cnecs = array(doc, "angleCnecs").len();
    report.voltage_cnecs = array(doc, "voltageCnecs").len();

    let mut network_actions = Vec::new();
    for a in array(doc, "networkActions") {
        match read_network_action(a, &index, &mut report)? {
            Some(action) => network_actions.push(action),
            None => {
                if let Some(id) = text(a, "id") {
                    report.dropped_actions.push(id);
                }
            }
        }
    }

    let mut range_actions = Vec::new();
    for (group, reader) in [
        ("pstRangeActions", read_pst as ActionReader),
        ("hvdcRangeActions", read_hvdc),
        ("injectionRangeActions", read_injection),
        ("counterTradeRangeActions", read_counter_trade),
    ] {
        for a in array(doc, group) {
            match read_range_action(a, &index, reader, &mut report)? {
                Some(action) => range_actions.push(action),
                None => {
                    if let Some(id) = text(a, "id") {
                        report.dropped_actions.push(id);
                    }
                }
            }
        }
    }

    let usage_limits = array(doc, "ra-usage-limits-per-instant")
        .iter()
        .filter_map(|l| {
            let instant = index.instant(&text(l, "instant")?)?;
            Some(RaUsageLimits {
                instant,
                max_ra: integer(l, "max-ra").map(|x| x as usize),
                max_tso: integer(l, "max-tso").map(|x| x as usize),
                max_topo_per_tso: per_tso(l, "max-topo-per-tso"),
                max_pst_per_tso: per_tso(l, "max-pst-per-tso"),
                max_ra_per_tso: per_tso(l, "max-ra-per-tso"),
                max_elementary_actions_per_tso: per_tso(l, "max-elementary-actions-per-tso"),
            })
        })
        .collect();

    let crac = Crac {
        id: text(doc, "id").unwrap_or_default(),
        name: text(doc, "name"),
        instants,
        contingencies,
        flow_cnecs,
        network_actions,
        range_actions,
        usage_limits,
        angle_cnecs: report.angle_cnecs,
        voltage_cnecs: report.voltage_cnecs,
    };
    Ok((crac, report))
}

fn per_tso(v: &Value, key: &str) -> HashMap<String, usize> {
    v.get(key)
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, x)| Some((k.clone(), x.as_u64()? as usize)))
                .collect()
        })
        .unwrap_or_default()
}

struct Index<'a> {
    instants: &'a [Instant],
    contingencies: &'a [Contingency],
}

impl Index<'_> {
    fn instant(&self, id: &str) -> Option<usize> {
        self.instants.iter().position(|i| i.id == id)
    }

    fn contingency(&self, id: &str) -> Option<usize> {
        self.contingencies.iter().position(|c| c.id == id)
    }
}

/// `1`/`2`, `"left"`/`"right"`, or absent — three spellings of the same field.
fn read_side(v: &Value) -> Side {
    match v.get("side") {
        Some(Value::Number(n)) => match n.as_i64() {
            Some(1) => Side::One,
            Some(2) => Side::Two,
            _ => Side::Both,
        },
        Some(Value::String(s)) => match s.as_str() {
            "left" => Side::One,
            "right" => Side::Two,
            _ => Side::Both,
        },
        _ => Side::Both,
    }
}

fn read_unit(v: &Value) -> Unit {
    match v.get("unit").and_then(Value::as_str).unwrap_or("").to_ascii_lowercase().as_str() {
        "megawatt" => Unit::Megawatt,
        "percent_imax" => Unit::PercentImax,
        "degree" => Unit::Degree,
        "kilovolt" => Unit::Kilovolt,
        _ => Unit::Ampere,
    }
}

fn read_flow_cnec(v: &Value, index: &Index<'_>) -> Result<Option<FlowCnec>, CracError> {
    let Some(id) = text(v, "id") else { return Ok(None) };
    let Some(element) = text(v, "networkElementId") else { return Ok(None) };
    let instant_id = text(v, "instant").unwrap_or_else(|| "preventive".into());
    let instant = index.instant(&instant_id).ok_or_else(|| CracError::UnknownInstant {
        referrer: id.clone(),
        instant: instant_id,
    })?;
    let contingency = match text(v, "contingencyId") {
        Some(c) => Some(index.contingency(&c).ok_or_else(|| CracError::UnknownContingency {
            referrer: id.clone(),
            contingency: c,
        })?),
        None => None,
    };

    let thresholds: Vec<Threshold> = array(v, "thresholds")
        .iter()
        .filter_map(|t| {
            let min = number(t, "min");
            let max = number(t, "max");
            (min.is_some() || max.is_some()).then_some(Threshold {
                unit: read_unit(t),
                min,
                max,
                side: read_side(t),
            })
        })
        .collect();

    let pair = |key: &str| -> Option<[Option<f64>; 2]> {
        let a = v.get(key)?.as_array()?;
        let get = |i: usize| a.get(i).and_then(Value::as_f64).filter(|x| x.is_finite());
        Some([get(0), get(1)])
    };

    Ok(Some(FlowCnec {
        id,
        network_element: element,
        state: State { instant, contingency },
        thresholds,
        reliability_margin: number(v, "reliabilityMargin").unwrap_or(0.0),
        optimized: v.get("optimized").and_then(Value::as_bool).unwrap_or(true),
        monitored: v.get("monitored").and_then(Value::as_bool).unwrap_or(false),
        operator: text(v, "operator"),
        i_max: pair("iMax"),
        nominal_v: pair("nominalV"),
    }))
}

fn read_usage_rules(
    v: &Value,
    index: &Index<'_>,
    report: &mut CracReport,
) -> Result<Vec<UsageRule>, CracError> {
    let mut rules = Vec::new();
    let instant_of = |v: &Value| -> Option<usize> {
        text(v, "instant").and_then(|i| index.instant(&i))
    };

    // Both spellings of "available at this instant".
    for key in ["onInstantUsageRules", "freeToUseUsageRules"] {
        for r in array(v, key) {
            if let Some(instant) = instant_of(r) {
                rules.push(UsageRule::OnInstant { instant });
            }
        }
    }
    // Both spellings of "available in this state".
    for key in ["onContingencyStateUsageRules", "onStateUsageRules"] {
        for r in array(v, key) {
            let (Some(instant), Some(c)) = (instant_of(r), text(r, "contingencyId")) else {
                continue;
            };
            if let Some(contingency) = index.contingency(&c) {
                rules.push(UsageRule::OnContingencyState {
                    state: State { instant, contingency: Some(contingency) },
                });
            }
        }
    }
    // Conditional on a CNEC. 2.x merged the three older per-kind spellings into
    // one, and names the CNEC by a key that varies with the kind.
    for key in [
        "onConstraintUsageRules",
        "onFlowConstraintUsageRules",
        "onAngleConstraintUsageRules",
        "onVoltageConstraintUsageRules",
    ] {
        for r in array(v, key) {
            let Some(instant) = instant_of(r) else { continue };
            let cnec = ["cnecId", "flowCnecId", "angleCnecId", "voltageCnecId"]
                .iter()
                .find_map(|k| text(r, k));
            if let Some(cnec) = cnec {
                rules.push(UsageRule::OnConstraint { instant, cnec });
            }
        }
    }
    for r in array(v, "onFlowConstraintInCountryUsageRules") {
        let (Some(instant), Some(country)) = (instant_of(r), text(r, "country")) else { continue };
        rules.push(UsageRule::OnFlowConstraintInCountry {
            instant,
            country,
            contingency: text(r, "contingencyId").and_then(|c| index.contingency(&c)),
        });
    }

    // Anything else ending in `UsageRules` is a spelling this reader has not
    // met. Counting it is the difference between a tolerant reader and one that
    // quietly makes a remedial action unavailable.
    if let Some(map) = v.as_object() {
        const KNOWN: [&str; 9] = [
            "onInstantUsageRules",
            "freeToUseUsageRules",
            "onContingencyStateUsageRules",
            "onStateUsageRules",
            "onConstraintUsageRules",
            "onFlowConstraintUsageRules",
            "onAngleConstraintUsageRules",
            "onVoltageConstraintUsageRules",
            "onFlowConstraintInCountryUsageRules",
        ];
        for (key, value) in map {
            if key.ends_with("UsageRules") && !KNOWN.contains(&key.as_str()) {
                let n = value.as_array().map(Vec::len).unwrap_or(0);
                *report.unknown_usage_rules.entry(key.clone()).or_insert(0) += n;
            }
        }
    }
    Ok(rules)
}

fn read_network_action(
    v: &Value,
    index: &Index<'_>,
    report: &mut CracReport,
) -> Result<Option<NetworkAction>, CracError> {
    let Some(id) = text(v, "id") else { return Ok(None) };
    let mut elementary = Vec::new();

    // `topologicalActions` is the 1.x spelling of `terminalsConnectionActions`.
    for key in ["terminalsConnectionActions", "topologicalActions"] {
        for a in array(v, key) {
            let Some(element) = text(a, "networkElementId") else { continue };
            let connected = matches!(
                text(a, "actionType").unwrap_or_default().to_ascii_lowercase().as_str(),
                "close"
            );
            elementary.push(ElementaryAction::TerminalsConnection { element, connected });
        }
    }
    for a in array(v, "switchActions") {
        let Some(element) = text(a, "networkElementId") else { continue };
        let open = matches!(
            text(a, "actionType").unwrap_or_default().to_ascii_lowercase().as_str(),
            "open"
        );
        elementary.push(ElementaryAction::Switch { element, open });
    }
    // `phaseTapChangerTapPositionActions` (2.x) and `pstSetpoints` (1.x) are
    // the same action; the older spelling calls the tap a "setpoint", which it
    // is not — it is an integer position, and reading it as an angle would put
    // a phase shifter tens of degrees away from where the CRAC meant.
    for (key, field) in
        [("phaseTapChangerTapPositionActions", "tapPosition"), ("pstSetpoints", "setpoint")]
    {
        for a in array(v, key) {
            let Some(element) = text(a, "networkElementId") else { continue };
            let Some(tap) = integer(a, field) else { continue };
            elementary.push(ElementaryAction::PstTapPosition { element, tap: tap as i32 });
        }
    }
    // `injectionSetpoints` is 1.x for what 2.x splits into generator and load
    // actions. It does not say which the element is, so it is read as a
    // generator set-point: both end up as an active-power target on a named
    // element, and the network decides what that element is.
    for a in array(v, "injectionSetpoints") {
        let Some(element) = text(a, "networkElementId") else { continue };
        let Some(p) = number(a, "setpoint") else { continue };
        elementary.push(ElementaryAction::GeneratorSetpoint { element, p });
    }
    for (key, is_load) in [("generatorActions", false), ("loadActions", true)] {
        for a in array(v, key) {
            let Some(element) = text(a, "networkElementId") else { continue };
            let Some(p) = number(a, "activePowerValue").or(number(a, "setpoint")) else { continue };
            elementary.push(if is_load {
                ElementaryAction::LoadSetpoint { element, p }
            } else {
                ElementaryAction::GeneratorSetpoint { element, p }
            });
        }
    }
    for a in array(v, "shuntCompensatorPositionActions") {
        let Some(element) = text(a, "networkElementId") else { continue };
        let Some(section) = integer(a, "sectionCount") else { continue };
        elementary.push(ElementaryAction::ShuntSection { element, section: section as i32 });
    }
    for a in array(v, "switchPairs") {
        let (Some(open), Some(close)) =
            (text(a, "open").or(text(a, "switchToOpen")), text(a, "close").or(text(a, "switchToClose")))
        else {
            continue;
        };
        elementary.push(ElementaryAction::SwitchPair { open, close });
    }

    // Spellings this reader does not know, counted rather than ignored.
    //
    // The suffix test has to cover `Setpoints` as well as `Actions`, because
    // the 1.x names do not end in `Actions` at all — `pstSetpoints` and
    // `injectionSetpoints` were invisible to an earlier version of this check,
    // which is exactly how 99 network actions came to be silently dropped
    // while the report said nothing was wrong.
    if let Some(map) = v.as_object() {
        const KNOWN: [&str; 10] = [
            "terminalsConnectionActions",
            "topologicalActions",
            "switchActions",
            "phaseTapChangerTapPositionActions",
            "pstSetpoints",
            "injectionSetpoints",
            "generatorActions",
            "loadActions",
            "shuntCompensatorPositionActions",
            "switchPairs",
        ];
        for (key, value) in map {
            let looks_like_an_action = key.ends_with("Actions") || key.ends_with("Setpoints");
            if looks_like_an_action && !KNOWN.contains(&key.as_str()) {
                let n = value.as_array().map(Vec::len).unwrap_or(0);
                *report.unknown_actions.entry(key.clone()).or_insert(0) += n;
            }
        }
    }

    if elementary.is_empty() {
        return Ok(None);
    }
    Ok(Some(NetworkAction {
        id,
        name: text(v, "name"),
        operator: text(v, "operator"),
        speed: integer(v, "speed"),
        activation_cost: number(v, "activationCost"),
        elementary,
        usage_rules: read_usage_rules(v, index, report)?,
    }))
}

type ActionReader = fn(&Value) -> Option<RangeActionKind>;

fn read_pst(v: &Value) -> Option<RangeActionKind> {
    let element = text(v, "networkElementId")?;
    let mut tap_to_angle: Vec<(i32, f64)> = v
        .get("tapToAngleConversionMap")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, x)| Some((k.parse::<i32>().ok()?, x.as_f64()?)))
                .collect()
        })
        .unwrap_or_default();
    // JSON object order is not tap order; the model requires ascending.
    tap_to_angle.sort_by_key(|(t, _)| *t);
    Some(RangeActionKind::Pst {
        element,
        initial_tap: integer(v, "initialTap").unwrap_or(0) as i32,
        tap_to_angle,
    })
}

fn read_hvdc(v: &Value) -> Option<RangeActionKind> {
    Some(RangeActionKind::Hvdc { element: text(v, "networkElementId")? })
}

fn read_injection(v: &Value) -> Option<RangeActionKind> {
    let distribution: Vec<(String, f64)> = array(v, "networkElementIdsAndKeys")
        .iter()
        .filter_map(|d| Some((text(d, "networkElementId")?, d.get("key")?.as_f64()?)))
        .collect();
    // Older files spell it as a plain object.
    let distribution = if distribution.is_empty() {
        v.get("networkElementIdsAndKeys")
            .and_then(Value::as_object)
            .map(|m| m.iter().filter_map(|(k, x)| Some((k.clone(), x.as_f64()?))).collect())
            .unwrap_or_default()
    } else {
        distribution
    };
    (!distribution.is_empty()).then_some(RangeActionKind::Injection { distribution })
}

fn read_counter_trade(v: &Value) -> Option<RangeActionKind> {
    Some(RangeActionKind::CounterTrade {
        exporting: text(v, "exportingCountry")?,
        importing: text(v, "importingCountry")?,
    })
}

fn read_range_action(
    v: &Value,
    index: &Index<'_>,
    reader: ActionReader,
    report: &mut CracReport,
) -> Result<Option<RangeAction>, CracError> {
    let Some(id) = text(v, "id") else { return Ok(None) };
    let Some(kind) = reader(v) else { return Ok(None) };
    let ranges: Vec<Range> = array(v, "ranges")
        .iter()
        .map(|r| Range {
            // Absent in older files, where every range was absolute.
            kind: match text(r, "rangeType").unwrap_or_default().as_str() {
                "relativeToInitialNetwork" => RangeKind::RelativeToInitialNetwork,
                "relativeToPreviousInstant" => RangeKind::RelativeToPreviousInstant,
                "relativeToPreviousTimeStep" => RangeKind::RelativeToPreviousTimeStep,
                _ => RangeKind::Absolute,
            },
            min: number(r, "min"),
            max: number(r, "max"),
        })
        .collect();
    Ok(Some(RangeAction {
        id,
        name: text(v, "name"),
        operator: text(v, "operator"),
        speed: integer(v, "speed"),
        activation_cost: number(v, "activationCost"),
        group: text(v, "groupId"),
        kind,
        ranges,
        usage_rules: read_usage_rules(v, index, report)?,
    }))
}
