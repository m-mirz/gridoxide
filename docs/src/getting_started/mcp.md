# MCP Server

`gridmcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server: it lets an LLM
chat — Claude Code, Claude Desktop, VS Code's agent mode, Codex, Cursor — read the power
networks in a directory and run gridoxide over them. Its source is `crates/gridoxide-mcp`.

It reads every format `gridoxide solve` reads: MATPOWER (`.m`), UCTE-DEF (`.uct`), PowSyBl IIDM
(`.xiidm`) and directories of CGMES profiles. The loading and the solve are the same code —
`gridoxide::load` — so the server and the CLI cannot disagree about a network.

## Tools

| Tool | |
|---|---|
| `list_networks` | the networks below a directory, with their format |
| `summary` | size, bus types, voltage levels, tap changers, regulating controls, areas, and the importer's notes |
| `power_flow` | an AC power flow, with the outer loops `gridoxide solve` offers as flags (`control_taps`, `enforce_q_limits`, `distribute_slack`, `area_interchange`, `control_remote_voltage`) |
| `results` | the buses or branches of the last `power_flow` on a network, filtered by name and sorted |

Every tool reads; none writes a file. Output is plain text sized for a model's context: long
lists stop at a `limit` and say how many rows are left.

Buses and branches carry the file's own names — MATPOWER bus numbers, UCTE node codes, IIDM
ids, CGMES `TopologicalNode` and equipment mRIDs. MATPOWER names no branches, so they are
`from-to` in bus numbers, with `#k` on the k-th parallel one. A bus's P and Q are its net
injection, worked out from the solved voltages by Kirchhoff, so they are right for slack and PV
buses too.

## Install

| | |
|---|---|
| VS Code | the [gridoxide extension](https://marketplace.visualstudio.com/items?itemName=m-mirz.gridoxide) bundles the binary and offers it to chat; nothing else to install |
| uv | `uvx --from gridoxide-mcp gridmcp` — no install step |
| pip / pipx | `pipx install gridoxide-mcp`, then `gridmcp` |
| cargo | `cargo install gridoxide-mcp` |

`--dir <path>` (else `GRIDOXIDE_NETWORK_DIR`, else the working directory) is where relative paths
resolve and where `list_networks` looks.

## Configure a client

Claude Code starts servers in the project directory, so `--dir` can be left out:

```bash
claude mcp add gridoxide -- uvx --from gridoxide-mcp gridmcp
```

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.gridoxide]
command = "uvx"
args = ["--from", "gridoxide-mcp", "gridmcp", "--dir", "/path/to/networks"]
```

Claude Desktop, Cursor, Windsurf and most others take an `mcpServers` block:

```json
{
  "mcpServers": {
    "gridoxide": {
      "command": "uvx",
      "args": ["--from", "gridoxide-mcp", "gridmcp", "--dir", "/path/to/networks"]
    }
  }
}
```

## Trying it by hand

The transport is one JSON-RPC message per line on stdin and stdout:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"power_flow","arguments":{"path":"tests/data/benchmark-grids/matpower/case14.m"}}}' \
  | cargo run -q -p gridoxide-mcp -- --dir .
```
