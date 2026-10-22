# gridoxide for VS Code

Ask chat about the power networks in your workspace. The extension offers `gridmcp`, the
[gridoxide](https://github.com/m-mirz/gridoxide) MCP server, to Copilot agent mode and every
other MCP client in VS Code — no Python, no install beyond the extension: the server is bundled
for each platform.

It reads MATPOWER cases (`.m`), UCTE-DEF (`.uct`), PowSyBl IIDM (`.xiidm`) and directories of
CGMES profiles, and gives chat four tools:

| Tool | |
|---|---|
| `list_networks` | the networks in the workspace, with their format |
| `summary` | size, bus types, voltage levels, tap changers, areas, and what the importer noted |
| `power_flow` | an AC power flow (Newton-Raphson), optionally with tap control, reactive limits, distributed slack or area interchange: convergence, losses, slack injection, voltage extremes, largest flows |
| `results` | the buses or branches of the last power flow, filtered and sorted |

Ask, for instance, *"Which bus in case14 has the highest voltage, and which line carries the
most?"* or *"Does the MicroGrid model still converge with reactive limits enforced?"*

The server reads the first workspace folder. Every tool reads; none writes a file.

## Settings

| Setting | |
|---|---|
| `gridoxide.mcp.enabled` | offer the server to chat (default on) |
| `gridoxide.mcp.path` | a `gridmcp` binary to run instead of the bundled one |

Outside VS Code, the same server is on PyPI: `uvx --from gridoxide-mcp gridmcp`.
