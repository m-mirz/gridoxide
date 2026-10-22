//! `gridmcp`: a Model Context Protocol server for gridoxide over stdio, so an
//! LLM chat (VS Code, Claude Code, Claude Desktop, …) can find the power
//! networks of a workspace, summarise one and run an AC power flow over it.
//!
//! The transport is newline-delimited JSON-RPC 2.0 on stdin/stdout; the
//! server answers `initialize`, `ping`, `tools/list` and `tools/call` and
//! ignores notifications. Requests are handled one at a time, in order. Every
//! tool ([`tools`]) reads; none writes a file.
//!
//! `gridmcp [--dir <path>]`: `--dir` (else `GRIDOXIDE_NETWORK_DIR`, else the
//! working directory) is where relative `path` arguments resolve and where
//! `list_networks` searches.

mod session;
mod tools;

use std::io::{BufRead, Write};
use std::path::PathBuf;

use serde_json::{Value, json};

use session::Session;

/// Protocol revisions this server speaks, newest first.
const PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

const INSTRUCTIONS: &str = "\
Tools over power networks, solved by gridoxide. A network is a MATPOWER case (`.m`), a UCTE-DEF \
file (`.uct`), a PowSyBl IIDM file (`.xiidm`) or a directory of CGMES profiles (EQ, SSH, TP, SV, … \
of one model); `path` names it, relative to the workspace. Start with `list_networks` when you do \
not know where the data is, then `summary`.
- Buses and branches carry the file's own names: MATPOWER bus numbers, UCTE node codes, IIDM ids, \
  CGMES TopologicalNode / equipment names (the mRID where there is none; `results`' `text` matches \
  mRIDs too). Branches are lines first, then transformers.
- `power_flow` solves an AC power flow (Newton-Raphson) and reports convergence, the buses \
  furthest from nominal voltage and the most loaded branches. Options turn on the outer loops: \
  tap control, reactive limits, distributed slack, area interchange.
- `results` lists the buses or branches of the last `power_flow` on that network, filtered and \
  sorted. Run `power_flow` first.
- Voltages are per-unit and kV, angles degrees, powers MW / MVAr / MVA. A bus's P and Q are its \
  net injection (generation minus load); a branch's are what enters it at each end.";

fn main() {
    let mut dir: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dir" => dir = args.next().map(PathBuf::from),
            "--version" | "-V" => {
                println!("gridmcp {}", env!("CARGO_PKG_VERSION"));
                return;
            }
            _ => {
                eprintln!("usage: gridmcp [--dir <network directory>]");
                eprintln!("An MCP server over stdio; an MCP client starts it.");
                std::process::exit(1);
            }
        }
    }
    let dir = dir
        .or_else(|| {
            std::env::var_os("GRIDOXIDE_NETWORK_DIR")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    let mut session = Session::new(dir);

    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => handle(&mut session, &msg),
            Err(e) => Some(error(Value::Null, -32700, &format!("parse error: {e}"))),
        };
        if let Some(reply) = reply {
            // serde_json writes no newlines inside a value, so one line is one message.
            let ok = serde_json::to_writer(&mut out, &reply).is_ok()
                && writeln!(out).is_ok()
                && out.flush().is_ok();
            if !ok {
                break;
            }
        }
    }
}

/// The reply to one message; `None` for a notification or a response.
fn handle(session: &mut Session, msg: &Value) -> Option<Value> {
    let method = msg.get("method").and_then(Value::as_str);
    let id = msg.get("id").cloned();
    let (Some(method), Some(id)) = (method, id) else {
        return None;
    };
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => {
            let asked = params.get("protocolVersion").and_then(Value::as_str);
            let version = asked
                .filter(|v| PROTOCOL_VERSIONS.contains(v))
                .unwrap_or(PROTOCOL_VERSIONS[0]);
            json!({
                "protocolVersion": version,
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "gridmcp", "version": env!("CARGO_PKG_VERSION") },
                "instructions": INSTRUCTIONS,
            })
        }
        "ping" => json!({}),
        "tools/list" => json!({ "tools": tools::definitions() }),
        "tools/call" => {
            let Some(name) = params.get("name").and_then(Value::as_str) else {
                return Some(error(id, -32602, "tools/call needs a tool name"));
            };
            let args = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match tools::call(session, name, &args) {
                None => return Some(error(id, -32602, &format!("unknown tool: {name}"))),
                // A failing tool is a result the model reads, not a protocol error.
                Some(Ok(text)) => {
                    json!({ "content": [{ "type": "text", "text": text }], "isError": false })
                }
                Some(Err(text)) => {
                    json!({ "content": [{ "type": "text", "text": text }], "isError": true })
                }
            }
        }
        "resources/list" => json!({ "resources": [] }),
        "prompts/list" => json!({ "prompts": [] }),
        _ => return Some(error(id, -32601, &format!("method not found: {method}"))),
    };
    Some(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}
