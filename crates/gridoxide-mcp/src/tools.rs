//! The tools: their definitions as `tools/list` returns them, and what each
//! does. Every tool reads; none writes a file. Output is plain text sized for
//! a model's context — long lists stop at a `limit` and say how many more
//! there are.

use std::fmt::Write as _;

use gridoxide::PowerFlowReport;
use gridoxide::branch_flow::{Terminal, branch_params, terminal_flow};
use gridoxide::load::{Network, SolveSettings};
use gridoxide::solver::{IslandStatus, SolveStatus};
use gridoxide::types::BusType;
use num_complex::Complex;
use serde_json::{Value, json};

use crate::session::{self, Session};

const PATH_DOC: &str = "The network, relative to the workspace: a MATPOWER `.m`, UCTE-DEF `.uct` or IIDM `.xiidm` file, \
or a directory of CGMES profile `.xml` files (one profile file stands for its directory).";

pub fn definitions() -> Value {
    let path = json!({ "type": "string", "description": PATH_DOC });
    let limit = |default: u64| json!({ "type": "integer", "minimum": 1, "description": format!("Most rows to return (default {default}).") });
    let read_only = json!({ "readOnlyHint": true, "openWorldHint": false });
    json!([
        {
            "name": "list_networks",
            "title": "List networks",
            "description": "Find the power networks below a directory: MATPOWER cases, UCTE-DEF and IIDM files, and directories of CGMES profiles.",
            "inputSchema": { "type": "object", "properties": {
                "root": { "type": "string", "description": "Directory to search, relative to the workspace (default: the workspace)." },
                "depth": { "type": "integer", "minimum": 0, "description": "How many directory levels to descend (default 6)." },
                "limit": limit(100),
            } },
            "annotations": read_only,
        },
        {
            "name": "summary",
            "title": "Summarise a network",
            "description": "A network's size (buses, lines, transformers, shunts), bus types, voltage levels, tap changers, \
    regulating controls, control areas, scheduled injections and what the importer noted while reading it.",
            "inputSchema": { "type": "object", "properties": { "path": path }, "required": ["path"] },
            "annotations": read_only,
        },
        {
            "name": "power_flow",
            "title": "Run an AC power flow",
            "description": "Solve an AC power flow (Newton-Raphson) and report convergence, islands, what the outer loops did, \
    losses, the slack buses' injections, the buses furthest from nominal voltage and the most loaded branches. \
    The solution is kept for `results`.",
            "inputSchema": { "type": "object", "properties": {
                "path": path,
                "control_taps": { "type": "boolean", "description": "Move on-load tap changers to hold their voltage or flow targets (default false)." },
                "enforce_q_limits": { "type": "boolean", "description": "Switch generators that hit their reactive limits from PV to PQ (default false)." },
                "distribute_slack": { "type": "boolean", "description": "Share the active imbalance across generators instead of one slack bus (default false)." },
                "area_interchange": { "type": "boolean", "description": "Hold each control area's net export at its schedule (CGMES ControlArea, UCTE country codes; default false). Not with distribute_slack." },
                "control_remote_voltage": { "type": "boolean", "description": "Put a machine regulating a remote bus's voltage on that machine (CGMES only; default false)." },
                "tol": { "type": "number", "exclusiveMinimum": 0, "description": "Mismatch tolerance, per-unit (default 1e-8)." },
                "max_iter": { "type": "integer", "minimum": 1, "description": "Newton iterations per solve (default 30)." },
                "limit": limit(5),
            }, "required": ["path"] },
            "annotations": read_only,
        },
        {
            "name": "results",
            "title": "Power flow results",
            "description": "The buses or branches of the last `power_flow` on a network: per bus voltage (pu, kV), angle and net injection; \
    per branch the flow entering it at each end, its larger apparent power and its losses. Filter by name, sort, limit.",
            "inputSchema": { "type": "object", "properties": {
                "path": path,
                "kind": { "type": "string", "enum": ["buses", "branches"], "description": "Which table." },
                "text": { "type": "string", "description": "Only rows whose name or id (CGMES mRID) contains this text (case-insensitive); for branches, either end's bus matches too." },
                "sort": { "type": "string", "enum": ["index", "deviation", "lowest", "highest", "loading", "losses"],
                    "description": "Buses: `index` (default), `deviation` from 1 pu, `lowest` or `highest` voltage. Branches: `index` (default), `loading` (largest MVA first), `losses`." },
                "limit": limit(50),
            }, "required": ["path", "kind"] },
            "annotations": read_only,
        },
    ])
}

/// `None` for an unknown tool; otherwise the text, or the error to show.
pub fn call(session: &mut Session, name: &str, args: &Value) -> Option<Result<String, String>> {
    Some(match name {
        "list_networks" => list_networks(session, args),
        "summary" => summary(session, args),
        "power_flow" => power_flow(session, args),
        "results" => results(session, args),
        _ => return None,
    })
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

fn bool_arg(args: &Value, key: &str) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn usize_arg(args: &Value, key: &str, default: usize) -> usize {
    args.get(key)
        .and_then(Value::as_u64)
        .map_or(default, |v| v as usize)
}

/// The `limit`-th row is not shown: say how many more there are.
fn more(out: &mut String, shown: usize, total: usize, hint: &str) {
    if total > shown {
        let _ = writeln!(out, "… and {} more{hint}", total - shown);
    }
}

fn size(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / (1u64 << 20) as f64),
        b if b >= 1 << 10 => format!("{:.0} kB", b as f64 / 1024.0),
        b => format!("{b} B"),
    }
}

fn list_networks(session: &mut Session, args: &Value) -> Result<String, String> {
    let root = session.resolve(str_arg(args, "root"));
    if !root.is_dir() {
        return Err(format!("{} is not a directory", root.display()));
    }
    let depth = usize_arg(args, "depth", 6);
    let limit = usize_arg(args, "limit", 100).max(1);
    let found = session::find_networks(&root, depth);
    if found.is_empty() {
        return Ok(format!(
            "No networks in {} ({} levels down). gridmcp reads MATPOWER .m, UCTE-DEF .uct, IIDM .xiidm and directories of CGMES profiles.",
            session.relative(&root),
            depth
        ));
    }
    let mut out = format!(
        "{} network(s) in {}:\n",
        found.len(),
        session.relative(&root)
    );
    for f in found.iter().take(limit) {
        let files = if f.files > 1 {
            format!("{} files, ", f.files)
        } else {
            String::new()
        };
        let _ = writeln!(
            out,
            "{}  {} ({files}{})",
            session.relative(&f.path),
            f.format.name(),
            size(f.bytes)
        );
    }
    more(
        &mut out,
        limit,
        found.len(),
        " (raise `limit`, or narrow `root`)",
    );
    Ok(out)
}

fn kv(net: &Network, bus: usize) -> f64 {
    net.buses[bus].u_rated / 1e3
}

fn summary(session: &mut Session, args: &Value) -> Result<String, String> {
    let loaded = session.network(str_arg(args, "path").unwrap_or(""))?;
    let net = &loaded.net;
    let base_mva = net.s_base_va / 1e6;
    let mut out = format!("{} — {}\n", loaded.name, net.format.name());
    let _ = writeln!(
        out,
        "{} bus(es), {} line(s), {} transformer(s), {} shunt(s); base {} MVA",
        net.buses.len(),
        net.lines.len(),
        net.transformers.len(),
        net.shunts.len(),
        base_mva
    );
    let count = |t: BusType| net.buses.iter().filter(|b| b.bus_type == t).count();
    let _ = writeln!(
        out,
        "bus types: {} slack, {} PV, {} PQ",
        count(BusType::Slack),
        count(BusType::PV),
        count(BusType::PQ)
    );

    // Rounded to 0.1 kV: nominal voltages written as 380 and 380.0000001 are one level.
    let mut levels: Vec<(i64, usize)> = Vec::new();
    for b in 0..net.buses.len() {
        let key = (kv(net, b) * 10.0).round() as i64;
        match levels.iter_mut().find(|(k, _)| *k == key) {
            Some((_, n)) => *n += 1,
            None => levels.push((key, 1)),
        }
    }
    levels.sort_by_key(|l| std::cmp::Reverse(l.0));
    let levels: Vec<String> = levels
        .iter()
        .map(|(k, n)| {
            if *k == 0 {
                format!("unstated: {n}")
            } else {
                format!("{} kV: {n}", *k as f64 / 10.0)
            }
        })
        .collect();
    let _ = writeln!(out, "buses by nominal voltage: {}", levels.join(", "));

    let (injected, withdrawn): (f64, f64) = net
        .buses
        .iter()
        .filter(|b| b.bus_type != BusType::Slack)
        .fold((0.0, 0.0), |(g, l), b| {
            if b.p_spec >= 0.0 {
                (g + b.p_spec, l)
            } else {
                (g, l - b.p_spec)
            }
        });
    let _ = writeln!(
        out,
        "scheduled net injection at non-slack buses: {:.1} MW in, {:.1} MW out (the slack takes the balance and the losses)",
        injected * base_mva,
        withdrawn * base_mva
    );
    let _ = writeln!(
        out,
        "{} tap changer(s), {} regulating tap control(s), {} regulating machine(s)",
        net.tap_changers.iter().filter(|c| c.is_some()).count(),
        net.regulation.len(),
        net.machines.len()
    );
    let areas = &net.areas.1;
    if !areas.is_empty() {
        let shown: Vec<&str> = areas.iter().take(20).map(String::as_str).collect();
        let _ = writeln!(
            out,
            "{} area(s){}: {}{}",
            areas.len(),
            if net.area_targets.is_empty() {
                " (no schedule)"
            } else {
                ""
            },
            shown.join(", "),
            if areas.len() > shown.len() {
                ", …"
            } else {
                ""
            }
        );
    }
    for note in &net.notes {
        let _ = writeln!(out, "note: {note}");
    }
    Ok(out)
}

/// One branch's flows, per-unit, entering it at each end.
struct Flow {
    from: usize,
    to: usize,
    pf: f64,
    qf: f64,
    pt: f64,
    qt: f64,
}

impl Flow {
    fn s_max(&self) -> f64 {
        self.pf.hypot(self.qf).max(self.pt.hypot(self.qt))
    }
    fn loss(&self) -> f64 {
        self.pf + self.pt
    }
}

/// Branch flows and each bus's net injection (per-unit), from the solved
/// voltages. Injection is what leaves a bus into its branches and shunts —
/// generation minus load, by Kirchhoff — so it is right for slack and PV
/// buses too, whose injection the solve determines.
fn flows(net: &Network, report: &PowerFlowReport) -> (Vec<Flow>, Vec<(f64, f64)>) {
    // Tap control moves taps: the flows are those of the transformers solved.
    let transformers = report
        .outer
        .as_ref()
        .map(|o| &o.transformers)
        .filter(|t| t.len() == net.transformers.len())
        .unwrap_or(&net.transformers);
    let v: Vec<Complex<f64>> = report
        .buses
        .iter()
        .map(|b| Complex::from_polar(b.voltage_mag, b.voltage_ang))
        .collect();
    let mut inj = vec![(0.0, 0.0); v.len()];
    let flows: Vec<Flow> = branch_params(&net.lines, transformers)
        .iter()
        .map(|bp| {
            let (pf, qf) = terminal_flow(bp, Terminal::From, &v);
            let (pt, qt) = terminal_flow(bp, Terminal::To, &v);
            inj[bp.from].0 += pf;
            inj[bp.from].1 += qf;
            inj[bp.to].0 += pt;
            inj[bp.to].1 += qt;
            Flow {
                from: bp.from,
                to: bp.to,
                pf,
                qf,
                pt,
                qt,
            }
        })
        .collect();
    for s in &net.shunts {
        // S = |V|² · conj(y)
        let vv = v[s.at].norm_sqr();
        inj[s.at].0 += vv * s.y.re;
        inj[s.at].1 -= vv * s.y.im;
    }
    (flows, inj)
}

fn label(labels: &[String], i: usize) -> String {
    match labels.get(i).filter(|l| !l.is_empty()) {
        Some(l) => l.clone(),
        None => format!("#{i}"),
    }
}

fn status_line(report: &PowerFlowReport) -> String {
    let s = &report.stats;
    // Solves through the outer loops keep no mismatch history; their passes
    // are reported with the loops.
    let trace = if s.iterations() == 0 {
        String::new()
    } else {
        format!(
            " after {} iteration(s), final mismatch {:.1e} pu",
            s.iterations(),
            s.final_mismatch()
        )
    };
    match s.status {
        SolveStatus::Converged => format!("converged{trace}"),
        other => format!(
            "did NOT converge ({other:?}){trace} — the voltages below are the last iterate, not a solution"
        ),
    }
}

fn power_flow(session: &mut Session, args: &Value) -> Result<String, String> {
    let defaults = SolveSettings::default();
    let settings = SolveSettings {
        control_taps: bool_arg(args, "control_taps"),
        enforce_q_limits: bool_arg(args, "enforce_q_limits"),
        distribute_slack: bool_arg(args, "distribute_slack"),
        area_interchange: bool_arg(args, "area_interchange"),
        control_remote_voltage: bool_arg(args, "control_remote_voltage"),
        tol: args
            .get("tol")
            .and_then(Value::as_f64)
            .filter(|t| *t > 0.0)
            .unwrap_or(defaults.tol),
        max_iter: usize_arg(args, "max_iter", defaults.max_iter).max(1),
        ..defaults
    };
    settings.check().map_err(|e| {
        e.replace("--distribute-slack", "distribute_slack")
            .replace("--area-interchange", "area_interchange")
    })?;
    let limit = usize_arg(args, "limit", 5).max(1);

    let loaded = session.network(str_arg(args, "path").unwrap_or(""))?;
    let report = loaded
        .net
        .solve(&settings)
        .map_err(|e| e.replace("--area-interchange", "area_interchange"))?;
    let net = &loaded.net;
    let base_mva = net.s_base_va / 1e6;
    let mut out = format!(
        "{} — {}: {} bus(es), {} line(s), {} transformer(s)\n{}\n",
        loaded.name,
        net.format.name(),
        net.buses.len(),
        net.lines.len(),
        net.transformers.len(),
        status_line(&report)
    );

    // Real networks come with a long tail of de-energized one-bus islands, so
    // the interesting ones are listed and the rest counted, as the CLI does.
    if report.islands.len() > 1 {
        let (mut trivial, mut trivial_buses) = (0, 0);
        for (i, island) in report.islands.iter().enumerate() {
            if island.status == IslandStatus::NoReferenceBus && island.bus_indices.len() <= 2 {
                trivial += 1;
                trivial_buses += island.bus_indices.len();
                continue;
            }
            let _ = writeln!(
                out,
                "island {i}: {} bus(es), {:?}",
                island.bus_indices.len(),
                island.status
            );
        }
        if trivial > 0 {
            let _ = writeln!(
                out,
                "… and {trivial} de-energized island(s) covering {trivial_buses} bus(es)"
            );
        }
    }

    if let Some(outer) = report.outer.as_ref() {
        if let Some(r) = outer.report.as_ref() {
            let loops: Vec<String> = r
                .loops
                .iter()
                .map(|l| format!("{} {}× {:?}", l.name, l.iterations, l.status))
                .collect();
            let _ = writeln!(
                out,
                "outer loops: {} re-solve(s), converged = {}{}; {}",
                r.total_iterations,
                r.converged,
                if r.budget_exhausted {
                    " (budget exhausted)"
                } else {
                    ""
                },
                loops.join(", ")
            );
        }
        if !outer.q_limit_switches.is_empty() {
            let names: Vec<String> = outer
                .q_limit_switches
                .iter()
                .take(limit)
                .map(|&b| label(&net.labels, b))
                .collect();
            let _ = writeln!(
                out,
                "{} bus(es) switched PV → PQ on a reactive limit: {}{}",
                outer.q_limit_switches.len(),
                names.join(", "),
                if outer.q_limit_switches.len() > limit {
                    ", …"
                } else {
                    ""
                }
            );
        }
        if let Some(s) = outer.slack.as_ref() {
            let _ = writeln!(
                out,
                "slack distribution: {:.3} MW moved over {} pass(es), converged = {}",
                s.shift.iter().sum::<f64>() * base_mva,
                s.outer_iterations,
                s.converged
            );
        }
        if let Some(a) = outer.area.as_ref() {
            let _ = writeln!(
                out,
                "area interchange over {} pass(es), converged = {}:",
                a.outer_iterations, a.converged
            );
            for (i, name) in net.areas.1.iter().enumerate().take(limit.max(10)) {
                let _ = writeln!(
                    out,
                    "  {name}: {:.3} MW exported, {:.3} MW off schedule{}",
                    a.interchange.get(i).copied().unwrap_or(0.0) * base_mva,
                    a.residual.get(i).copied().unwrap_or(0.0) * base_mva,
                    if a.dependent == Some(i) {
                        " (dependent: takes the residual)"
                    } else {
                        ""
                    }
                );
            }
        }
        if !outer.taps.is_empty() {
            let moved = outer
                .taps
                .iter()
                .filter(|c| c.final_position != c.initial_position)
                .count();
            let _ = writeln!(
                out,
                "tap controllers: {} of {} moved",
                moved,
                outer.taps.len()
            );
            for c in outer
                .taps
                .iter()
                .filter(|c| c.final_position != c.initial_position)
                .take(limit)
            {
                let _ = writeln!(
                    out,
                    "  {} {} → {} {:?}",
                    c.id, c.initial_position, c.final_position, c.outcome
                );
            }
        }
        if !outer.remote.is_empty() {
            let _ = writeln!(
                out,
                "remote voltage control: {} machine(s)",
                outer.remote.len()
            );
        }
    }

    let (flows, inj) = flows(net, &report);
    let losses: f64 = flows.iter().map(Flow::loss).sum();
    let _ = writeln!(out, "losses: {:.3} MW", losses * base_mva);
    let slacks: Vec<usize> = report
        .buses
        .iter()
        .enumerate()
        .filter(|(_, b)| b.bus_type == BusType::Slack && b.voltage_mag > 0.0)
        .map(|(i, _)| i)
        .collect();
    for &i in slacks.iter().take(limit) {
        let _ = writeln!(
            out,
            "slack {}: P {:.2} MW, Q {:.2} MVAr",
            label(&net.labels, i),
            inj[i].0 * base_mva,
            inj[i].1 * base_mva
        );
    }
    more(&mut out, limit, slacks.len(), " slack bus(es)");

    let mut extremes: Vec<usize> = (0..report.buses.len())
        .filter(|&i| report.buses[i].voltage_mag > 0.0)
        .collect();
    extremes.sort_by(|&a, &b| {
        let d = |i: usize| (report.buses[i].voltage_mag - 1.0).abs();
        d(b).total_cmp(&d(a))
    });
    if !extremes.is_empty() {
        let _ = writeln!(out, "furthest from nominal:");
        for &i in extremes.iter().take(limit) {
            let b = &report.buses[i];
            let _ = writeln!(
                out,
                "  {}  {:.4} pu{}",
                label(&net.labels, i),
                b.voltage_mag,
                if b.u_rated > 0.0 {
                    format!(" ({:.2} kV)", b.voltage_mag * b.u_rated / 1e3)
                } else {
                    String::new()
                }
            );
        }
    }

    let mut loaded_branches: Vec<usize> = (0..flows.len()).collect();
    loaded_branches.sort_by(|&a, &b| flows[b].s_max().total_cmp(&flows[a].s_max()));
    if !loaded_branches.is_empty() {
        let _ = writeln!(out, "largest branch flows:");
        for &k in loaded_branches.iter().take(limit) {
            let f = &flows[k];
            let _ = writeln!(
                out,
                "  {} ({} → {})  {:.2} MVA, P {:.2} MW",
                label(&net.branch_labels, k),
                label(&net.labels, f.from),
                label(&net.labels, f.to),
                f.s_max() * base_mva,
                f.pf * base_mva
            );
        }
    }
    let _ = writeln!(out, "`results` lists every bus or branch of this solution.");
    loaded.solved = Some((settings, report));
    Ok(out)
}

fn results(session: &mut Session, args: &Value) -> Result<String, String> {
    let kind = str_arg(args, "kind").unwrap_or("");
    if kind != "buses" && kind != "branches" {
        return Err("`kind` is `buses` or `branches`".into());
    }
    let sort = str_arg(args, "sort").unwrap_or("index");
    let limit = usize_arg(args, "limit", 50).max(1);
    let text = str_arg(args, "text")
        .map(|t| t.trim().to_lowercase())
        .filter(|t| !t.is_empty());
    let matches = |l: &str| text.as_ref().is_none_or(|t| l.to_lowercase().contains(t));

    let loaded = session.network(str_arg(args, "path").unwrap_or(""))?;
    let Some((_, report)) = loaded.solved.as_ref() else {
        return Err(format!(
            "no power flow solved over {} yet (or its files changed since): run `power_flow` first",
            loaded.name
        ));
    };
    let net = &loaded.net;
    let base_mva = net.s_base_va / 1e6;
    let (flows, inj) = flows(net, report);
    let mut out = format!("{} — {}\n", loaded.name, status_line(report));

    if kind == "buses" {
        let mut rows: Vec<usize> = (0..report.buses.len())
            .filter(|&i| matches(&label(&net.labels, i)) || matches(&label(&net.bus_ids, i)))
            .collect();
        let v = |i: usize| report.buses[i].voltage_mag;
        match sort {
            "index" => {}
            "deviation" => rows.sort_by(|&a, &b| (v(b) - 1.0).abs().total_cmp(&(v(a) - 1.0).abs())),
            "lowest" => rows.sort_by(|&a, &b| v(a).total_cmp(&v(b))),
            "highest" => rows.sort_by(|&a, &b| v(b).total_cmp(&v(a))),
            other => {
                return Err(format!(
                    "buses sort by `index`, `deviation`, `lowest` or `highest`, not `{other}`"
                ));
            }
        }
        let _ = writeln!(
            out,
            "{} bus(es){}; index  name  type  V pu  V kV  angle °  P MW  Q MVAr",
            rows.len(),
            if text.is_some() { " matching" } else { "" }
        );
        for &i in rows.iter().take(limit) {
            let b = &report.buses[i];
            let kv = if b.u_rated > 0.0 {
                format!("{:.2}", b.voltage_mag * b.u_rated / 1e3)
            } else {
                "-".into()
            };
            let _ = writeln!(
                out,
                "{i}  {}  {:?}  {:.4}  {kv}  {:.3}  {:.2}  {:.2}",
                label(&net.labels, i),
                b.bus_type,
                b.voltage_mag,
                b.voltage_ang.to_degrees(),
                inj[i].0 * base_mva,
                inj[i].1 * base_mva
            );
        }
        more(
            &mut out,
            limit,
            rows.len(),
            " (raise `limit`, or filter with `text`)",
        );
    } else {
        let mut rows: Vec<usize> = (0..flows.len())
            .filter(|&k| {
                let f = &flows[k];
                [(&net.branch_labels, k), (&net.branch_ids, k)]
                    .into_iter()
                    .chain([&net.labels, &net.bus_ids].map(|l| (l, f.from)))
                    .chain([&net.labels, &net.bus_ids].map(|l| (l, f.to)))
                    .any(|(l, i)| matches(&label(l, i)))
            })
            .collect();
        match sort {
            "index" => {}
            "loading" => rows.sort_by(|&a, &b| flows[b].s_max().total_cmp(&flows[a].s_max())),
            "losses" => rows.sort_by(|&a, &b| flows[b].loss().total_cmp(&flows[a].loss())),
            other => {
                return Err(format!(
                    "branches sort by `index`, `loading` or `losses`, not `{other}`"
                ));
            }
        }
        let _ = writeln!(
            out,
            "{} branch(es){} ({} line(s) first, then transformers); index  name  from → to  P from MW  Q from MVAr  P to MW  Q to MVAr  max MVA  loss MW",
            rows.len(),
            if text.is_some() { " matching" } else { "" },
            net.lines.len()
        );
        for &k in rows.iter().take(limit) {
            let f = &flows[k];
            let _ = writeln!(
                out,
                "{k}  {}  {} → {}  {:.2}  {:.2}  {:.2}  {:.2}  {:.2}  {:.3}",
                label(&net.branch_labels, k),
                label(&net.labels, f.from),
                label(&net.labels, f.to),
                f.pf * base_mva,
                f.qf * base_mva,
                f.pt * base_mva,
                f.qt * base_mva,
                f.s_max() * base_mva,
                f.loss() * base_mva
            );
        }
        more(
            &mut out,
            limit,
            rows.len(),
            " (raise `limit`, or filter with `text`)",
        );
    }
    Ok(out)
}
