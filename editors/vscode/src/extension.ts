// Offers `gridmcp` to chat as an MCP server, reading the networks of the
// first workspace folder. VS Code starts it when a chat first uses it; the
// extension itself does nothing else.

import * as fs from "fs";
import * as path from "path";
import { EventEmitter, ExtensionContext, lm, McpStdioServerDefinition, workspace } from "vscode";

/** The `gridoxide.mcp.path` setting, else the bundled binary, else `PATH`. */
function serverCommand(context: ExtensionContext): string {
  const configured = workspace.getConfiguration("gridoxide").get<string>("mcp.path");
  if (configured) {
    return configured;
  }
  const exe = process.platform === "win32" ? "gridmcp.exe" : "gridmcp";
  const bundled = context.asAbsolutePath(path.join("server", exe));
  if (!fs.existsSync(bundled)) {
    return exe;
  }
  if (process.platform !== "win32") {
    // A VSIX is a zip; not every unpacker keeps the executable bit.
    try {
      fs.chmodSync(bundled, 0o755);
    } catch {
      // Read-only install: run it as it is.
    }
  }
  return bundled;
}

/** The server definitions VS Code is offered: none when disabled. */
function definitions(context: ExtensionContext): McpStdioServerDefinition[] {
  if (!workspace.getConfiguration("gridoxide").get<boolean>("mcp.enabled", true)) {
    return [];
  }
  const folder = workspace.workspaceFolders?.[0]?.uri.fsPath;
  const args = folder ? ["--dir", folder] : [];
  const version = context.extension.packageJSON.version as string;
  return [new McpStdioServerDefinition("gridoxide", serverCommand(context), args, {}, version)];
}

/** What `activate` returns, for the extension's tests (test/suite.ts). */
export interface Api {
  definitions(): McpStdioServerDefinition[];
}

export function activate(context: ExtensionContext): Api {
  const changed = new EventEmitter<void>();
  context.subscriptions.push(
    changed,
    lm.registerMcpServerDefinitionProvider("gridoxide", {
      onDidChangeMcpServerDefinitions: changed.event,
      provideMcpServerDefinitions: () => definitions(context),
    }),
    workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("gridoxide.mcp")) {
        changed.fire();
      }
    }),
    workspace.onDidChangeWorkspaceFolders(() => changed.fire()),
  );
  return { definitions: () => definitions(context) };
}

export function deactivate(): void {}
