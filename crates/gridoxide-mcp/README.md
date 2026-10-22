# gridoxide-mcp

`gridmcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server for power
networks, built on [gridoxide](https://github.com/m-mirz/gridoxide). It lets an LLM chat answer
questions about a network: how big it is, whether it solves, which buses sit furthest from
nominal voltage and which branches carry the most.

It reads MATPOWER cases (`.m`), UCTE-DEF (`.uct`), PowSyBl IIDM (`.xiidm`) and directories of
CGMES profiles. Every tool reads; none writes a file.

| Tool | |
|---|---|
| `list_networks` | the networks below a directory, with their format |
| `summary` | size, bus types, voltage levels, tap changers, areas, and what the importer noted |
| `power_flow` | an AC power flow (Newton-Raphson), optionally with tap control, reactive limits, distributed slack or area interchange: convergence, losses, slack injection, voltage extremes, largest flows |
| `results` | the buses or branches of the last `power_flow`, filtered and sorted |

`--dir <path>` (else `GRIDOXIDE_NETWORK_DIR`, else the working directory) is where relative
paths resolve.

## Install

With [uv](https://docs.astral.sh/uv/), nothing needs installing first; clients start it as

```
uvx --from gridoxide-mcp gridmcp --dir /path/to/networks
```

or install it once with `uv tool install gridoxide-mcp` (or `pipx install gridoxide-mcp`) and
run `gridmcp`. It is also on crates.io (`cargo install gridoxide-mcp`) and bundled in the
[VS Code extension](https://marketplace.visualstudio.com/items?itemName=m-mirz.gridoxide).

## Configure a client

Claude Code (servers start in the project directory, so `--dir` can be left out):

```
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

## Licence

gridmcp's own code is Apache-2.0. The binary links gridoxide, whose sparse solver is a
translation of SuiteSparse (BSD-3-Clause for AMD, LGPL-2.1-or-later for BTF/KLU); see
gridoxide's `docs/src/reference/provenance.md`.
