// Smoke test in a running VS Code (test/run.js): the server definition the
// extension offers, and the bundled binary started as VS Code starts it —
// command, arguments and all. What the server's own tests (crates/gridoxide-mcp)
// cannot see: where the extension finds the binary and what directory it hands it.

import * as assert from "assert";
import * as fs from "fs";
import * as path from "path";
import { spawn } from "child_process";
import { extensions } from "vscode";
import type { Api } from "../src/extension";

const data = process.env.GRIDOXIDE_TEST_DATA as string;

/** Sends `messages` to `command args` over stdio and collects one reply per request. */
function exchange(command: string, args: string[], messages: object[]): Promise<any[]> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["pipe", "pipe", "inherit"] });
    let out = "";
    const expected = messages.filter((m) => "id" in m).length;
    child.stdout.on("data", (chunk) => {
      out += chunk;
      const lines = out.split("\n").filter((l) => l.trim());
      if (lines.length >= expected) {
        child.kill();
        resolve(lines.map((l) => JSON.parse(l)));
      }
    });
    child.on("error", reject);
    for (const m of messages) child.stdin.write(JSON.stringify(m) + "\n");
  });
}

/** The entry point @vscode/test-electron calls. */
export async function run(): Promise<void> {
  const ext = extensions.getExtension<Api>("m-mirz.gridoxide");
  assert.ok(ext, "extension not found");
  const api = await ext.activate();

  const defs = api.definitions();
  assert.strictEqual(defs.length, 1);
  const def = defs[0];
  assert.strictEqual(def.label, "gridoxide");
  assert.strictEqual(path.dirname(def.command), path.join(ext.extensionPath, "server"), def.command);
  assert.ok(fs.existsSync(def.command), def.command);
  assert.deepStrictEqual(def.args, ["--dir", data]);

  const [init, list, call] = await exchange(def.command, def.args, [
    { jsonrpc: "2.0", id: 1, method: "initialize", params: { protocolVersion: "2025-06-18", capabilities: {} } },
    { jsonrpc: "2.0", method: "notifications/initialized" },
    { jsonrpc: "2.0", id: 2, method: "tools/list" },
    { jsonrpc: "2.0", id: 3, method: "tools/call", params: { name: "list_networks", arguments: {} } },
  ]);
  assert.strictEqual(init.result.serverInfo.name, "gridmcp");
  assert.ok(list.result.tools.some((t: { name: string }) => t.name === "power_flow"));
  assert.match(call.result.content[0].text, /case14\.m {2}MATPOWER/);
}

