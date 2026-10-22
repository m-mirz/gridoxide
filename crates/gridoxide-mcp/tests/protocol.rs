//! `gridmcp` driven over stdio, as an MCP client drives it, against the
//! benchmark-grids fixtures (and MicroGrid, where CGMES-Test-Configurations is
//! checked out — CI does not fetch it).

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

const CASE14: &str = "tests/data/benchmark-grids/matpower/case14.m";
const UCTE: &str =
    "tests/data/benchmark-grids/powsybl-open-rao/files/cases/epic13/TestCase12NodesForCurative.uct";
const IIDM: &str = "tests/data/benchmark-grids/powsybl-open-rao/converted/TestCase12Nodes.xiidm";
const MICROGRID: &str = "tests/data/CGMES-Test-Configurations/v3.0/MicroGrid/MicroGid-BaseCase/MicroGrid-BaseCase-Merged";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Server {
    fn start() -> Self {
        let case = repo_root().join(CASE14);
        assert!(
            case.is_file(),
            "missing test data {} — run `git submodule update --init tests/data/benchmark-grids`",
            case.display()
        );
        let mut child = Command::new(env!("CARGO_BIN_EXE_gridmcp"))
            .arg("--dir")
            .arg(repo_root())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("start gridmcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut server = Server {
            child,
            stdin,
            stdout,
            next_id: 1,
        };
        let init = server.request(
            "initialize",
            json!({ "protocolVersion": "2025-06-18", "capabilities": {} }),
        );
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(init["result"]["serverInfo"]["name"], "gridmcp");
        server.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));
        server
    }

    fn send(&mut self, msg: &Value) {
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }));
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        let reply: Value = serde_json::from_str(&line).expect("one JSON message per line");
        assert_eq!(reply["id"], id);
        reply
    }

    /// A tool's text, and whether it is an error.
    fn call(&mut self, name: &str, args: Value) -> (String, bool) {
        let reply = self.request("tools/call", json!({ "name": name, "arguments": args }));
        let result = &reply["result"];
        let text = result["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("{name}: no text in {reply}"))
            .to_string();
        (text, result["isError"].as_bool().unwrap())
    }

    fn ok(&mut self, name: &str, args: Value) -> String {
        let (text, is_error) = self.call(name, args);
        assert!(!is_error, "{name} failed: {text}");
        text
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn lists_its_tools() {
    let mut s = Server::start();
    let reply = s.request("tools/list", json!({}));
    let names: Vec<&str> = reply["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["list_networks", "summary", "power_flow", "results"]);
    assert_eq!(s.request("ping", json!({}))["result"], json!({}));
}

#[test]
fn protocol_errors() {
    let mut s = Server::start();
    let unknown = s.request("tools/call", json!({ "name": "nope", "arguments": {} }));
    assert_eq!(unknown["error"]["code"], -32602);
    let method = s.request("nope/nope", json!({}));
    assert_eq!(method["error"]["code"], -32601);
}

#[test]
fn finds_networks() {
    let mut s = Server::start();
    let text = s.ok(
        "list_networks",
        json!({ "root": "tests/data/benchmark-grids", "limit": 10000 }),
    );
    assert!(text.contains("matpower/case14.m  MATPOWER"), "{text}");
    assert!(
        text.contains("TestCase12NodesForCurative.uct  UCTE-DEF"),
        "{text}"
    );
    assert!(text.contains("TestCase12Nodes.xiidm  IIDM"), "{text}");
}

#[test]
fn summarises_case14() {
    let mut s = Server::start();
    let text = s.ok("summary", json!({ "path": CASE14 }));
    assert!(
        text.contains("14 bus(es), 17 line(s), 3 transformer(s), 1 shunt(s); base 100 MVA"),
        "{text}"
    );
    assert!(text.contains("1 slack, 4 PV, 9 PQ"), "{text}");
}

#[test]
fn solves_case14() {
    let mut s = Server::start();

    // Nothing to list before a solve.
    let (text, is_error) = s.call("results", json!({ "path": CASE14, "kind": "buses" }));
    assert!(
        is_error && text.contains("run `power_flow` first"),
        "{text}"
    );

    let text = s.ok("power_flow", json!({ "path": CASE14 }));
    assert!(text.contains("converged after"), "{text}");
    // MATPOWER's own solution of case14: slack 232.39 MW, losses 13.393 MW.
    assert!(text.contains("slack 1: P 232.39 MW"), "{text}");
    assert!(text.contains("losses: 13.393 MW"), "{text}");
    assert!(text.contains("1-2 (1 → 2)"), "{text}");

    let buses = s.ok(
        "results",
        json!({ "path": CASE14, "kind": "buses", "sort": "highest", "limit": 1 }),
    );
    assert!(buses.contains("\n7  8  PV  1.0900"), "{buses}");
    assert!(buses.contains("… and 13 more"), "{buses}");

    let branches = s.ok(
        "results",
        json!({ "path": CASE14, "kind": "branches", "text": "1-5" }),
    );
    assert!(branches.contains("1 branch(es) matching"), "{branches}");

    let (text, is_error) = s.call(
        "results",
        json!({ "path": CASE14, "kind": "buses", "sort": "loading" }),
    );
    assert!(is_error, "{text}");
}

#[test]
fn solves_ucte_and_iidm() {
    let mut s = Server::start();
    let text = s.ok(
        "power_flow",
        json!({ "path": UCTE, "area_interchange": true }),
    );
    assert!(text.contains("converged"), "{text}");
    assert!(text.contains("area interchange"), "{text}");
    let text = s.ok(
        "power_flow",
        json!({ "path": IIDM, "distribute_slack": true }),
    );
    assert!(text.contains("slack distribution"), "{text}");
}

#[test]
fn refuses_what_it_cannot_do() {
    let mut s = Server::start();
    let (text, is_error) = s.call("summary", json!({ "path": "no/such/case.m" }));
    assert!(is_error && text.contains("does not exist"), "{text}");
    let (text, is_error) = s.call("summary", json!({ "path": "Cargo.toml" }));
    assert!(is_error && text.contains("not a network"), "{text}");
    let (text, is_error) = s.call(
        "power_flow",
        json!({ "path": CASE14, "distribute_slack": true, "area_interchange": true }),
    );
    assert!(is_error && text.contains("cannot both be given"), "{text}");
    // MATPOWER states no areas to hold.
    let (text, is_error) = s.call(
        "power_flow",
        json!({ "path": CASE14, "area_interchange": true }),
    );
    assert!(
        is_error && text.contains("needs an area assignment"),
        "{text}"
    );
}

#[test]
fn solves_microgrid() {
    if !repo_root().join(MICROGRID).is_dir() {
        eprintln!("skipped: {MICROGRID} is not checked out");
        return;
    }
    let mut s = Server::start();
    let summary = s.ok("summary", json!({ "path": MICROGRID }));
    assert!(
        summary.contains("16 bus(es), 13 line(s), 9 transformer(s)"),
        "{summary}"
    );
    assert!(summary.contains("2 area(s): NL, BE"), "{summary}");
    let text = s.ok(
        "power_flow",
        json!({ "path": MICROGRID, "enforce_q_limits": true }),
    );
    assert!(text.contains("converged"), "{text}");
    assert!(text.contains("switched PV → PQ"), "{text}");
    // Buses and branches go by their IdentifiedObject.name ...
    let branches = s.ok(
        "results",
        json!({ "path": MICROGRID, "kind": "branches", "sort": "loading", "limit": 1 }),
    );
    assert!(branches.contains("NL_TR2_3  N1230822413 → "), "{branches}");
    assert!(!branches.contains("_80016742"), "{branches}");
    // ... and are still found by mRID.
    let by_mrid = s.ok(
        "results",
        json!({ "path": MICROGRID, "kind": "branches", "text": "80016742-31b3" }),
    );
    assert!(
        by_mrid.contains("1 branch(es) matching") && by_mrid.contains("NL_TR2_3"),
        "{by_mrid}"
    );
}
