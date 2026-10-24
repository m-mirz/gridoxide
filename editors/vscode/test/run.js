// Runs test/suite.ts in VS Code with this extension, on a workspace holding
// case14: `npm test`. VS Code is downloaded into .vscode-test unless
// GRIDOXIDE_TEST_VSCODE names an executable. Needs server/gridmcp (built and
// copied there, as for packaging) and a display (xvfb-run in CI).
const fs = require("fs");
const os = require("os");
const path = require("path");
const { runTests } = require("@vscode/test-electron");

async function main() {
  const root = path.resolve(__dirname, "..");
  const data = path.resolve(root, "../../tests/data/benchmark-grids/matpower/case14.m");
  if (!fs.existsSync(data)) throw new Error(`no test data at ${data}: git submodule update --init tests/data/benchmark-grids`);
  const server = path.join(root, "server", process.platform === "win32" ? "gridmcp.exe" : "gridmcp");
  if (!fs.existsSync(server)) throw new Error(`no ${server}: build gridmcp and copy it there`);
  const work = fs.mkdtempSync(path.join(os.tmpdir(), "gridoxide-test-"));
  fs.copyFileSync(data, path.join(work, "case14.m"));
  try {
    await runTests({
      vscodeExecutablePath: process.env.GRIDOXIDE_TEST_VSCODE || undefined,
      extensionDevelopmentPath: root,
      extensionTestsPath: path.join(root, "dist", "test", "suite.js"),
      launchArgs: [work, "--disable-extensions", "--disable-workspace-trust", "--skip-welcome", "--skip-release-notes"],
      extensionTestsEnv: { GRIDOXIDE_TEST_DATA: work },
    });
  } finally {
    fs.rmSync(work, { recursive: true, force: true });
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
