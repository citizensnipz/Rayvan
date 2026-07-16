#!/usr/bin/env node
/**
 * Prepare Tauri sidecar binaries for the current host triple.
 *
 * Builds `@rayvan/daemon` and `@rayvan/mcp-server`, then places host-triple
 * launchers under:
 *   apps/desktop/src-tauri/binaries/rayvand-<triple>[.exe]
 *   apps/desktop/src-tauri/binaries/rayvan-mcp-<triple>[.exe]
 *
 * Unix: bash wrappers that exec `node <dist/main.js>` (or tsx + source).
 * Windows: `.cmd` launchers always, plus a small .NET Framework launcher `.exe`
 * when PowerShell `Add-Type -OutputAssembly` is available. Empty build.rs
 * placeholders are replaced. Production installers should swap these for
 * Node SEA / pkg single-file binaries.
 *
 * Usage:
 *   pnpm prepare:sidecars
 *   pnpm --filter @rayvan/desktop prepare:sidecars
 *   node apps/desktop/scripts/prepare-sidecars.mjs
 *
 * Force rewrite even when a non-empty binary exists:
 *   RAYVAN_FORCE_SIDECAR_SCAFFOLD=1 pnpm prepare:sidecars
 */

import { spawnSync } from "node:child_process";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const desktopRoot = resolve(__dirname, "..");
const repoRoot = resolve(desktopRoot, "../..");
const binariesDir = join(desktopRoot, "src-tauri", "binaries");

function hostTriple() {
  const fromRustc = spawnSync("rustc", ["-vV"], { encoding: "utf8" });
  if (fromRustc.status === 0) {
    const match = /host:\s+(\S+)/.exec(fromRustc.stdout);
    if (match) return match[1];
  }
  const { platform, arch } = process;
  if (platform === "win32" && arch === "x64") return "x86_64-pc-windows-msvc";
  if (platform === "win32" && arch === "arm64") return "aarch64-pc-windows-msvc";
  if (platform === "darwin" && arch === "arm64") return "aarch64-apple-darwin";
  if (platform === "darwin" && arch === "x64") return "x86_64-apple-darwin";
  if (platform === "linux" && arch === "x64") return "x86_64-unknown-linux-gnu";
  if (platform === "linux" && arch === "arm64") return "aarch64-unknown-linux-gnu";
  throw new Error(`Unable to determine host triple for ${platform}/${arch}`);
}

function ensureBuilt(filter) {
  const result = spawnSync(
    "pnpm",
    ["--filter", filter, "build"],
    { cwd: repoRoot, stdio: "inherit", shell: process.platform === "win32" },
  );
  if (result.status !== 0) {
    console.warn(
      `Warning: failed to build ${filter} (exit ${result.status}). Falling back to source/tsx launchers when dist is missing.`,
    );
  }
}

function resolveEntry(kind) {
  const distMain =
    kind === "daemon"
      ? join(repoRoot, "apps/daemon/dist/main.js")
      : join(repoRoot, "apps/mcp-server/dist/main.js");
  if (existsSync(distMain)) {
    return { mode: "node", scriptPath: distMain };
  }

  const srcMain =
    kind === "daemon"
      ? join(repoRoot, "apps/daemon/src/main.ts")
      : join(repoRoot, "apps/mcp-server/src/main.ts");
  if (!existsSync(srcMain)) {
    throw new Error(`Missing both dist and source entry for ${kind}`);
  }

  const tsxCli = [
    join(
      repoRoot,
      `apps/${kind === "daemon" ? "daemon" : "mcp-server"}/node_modules/tsx/dist/cli.mjs`,
    ),
    join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
  ].find((candidate) => existsSync(candidate));
  if (!tsxCli) {
    throw new Error(
      `Missing ${distMain} and no tsx CLI for source fallback. Build ${kind} or install tsx.`,
    );
  }
  return { mode: "tsx", scriptPath: srcMain, tsxCli };
}

function nodeInvocation(entry) {
  if (entry.mode === "tsx") {
    return `node "${entry.tsxCli}" "${entry.scriptPath}"`;
  }
  return `node "${entry.scriptPath}"`;
}

function isPlaceholderOrMissing(targetPath) {
  if (!existsSync(targetPath)) return true;
  if (process.env.RAYVAN_FORCE_SIDECAR_SCAFFOLD === "1") return true;
  try {
    return statSync(targetPath).size === 0;
  } catch {
    return true;
  }
}

function writeUnixLauncher(targetPath, entry) {
  writeFileSync(
    targetPath,
    `#!/usr/bin/env bash\nexec ${nodeInvocation(entry)} "$@"\n`,
    { mode: 0o755 },
  );
  chmodSync(targetPath, 0o755);
  console.log(`Wrote launcher ${targetPath}`);
}

function writeWindowsCmd(cmdPath, entry) {
  writeFileSync(cmdPath, `@echo off\r\n${nodeInvocation(entry)} %*\r\n`, "utf8");
  console.log(`Wrote ${cmdPath}`);
}

/**
 * Compile a tiny .NET Framework console launcher that forwards to node.
 * Writes C# to a temp file to avoid PowerShell quoting pitfalls.
 */
function tryWriteWindowsDotNetLauncher(exePath, entry) {
  const tmpDir = join(binariesDir, ".scaffold-tmp");
  mkdirSync(tmpDir, { recursive: true });
  const csPath = join(tmpDir, `launcher-${Date.now()}.cs`);

  const argParts =
    entry.mode === "tsx"
      ? [entry.tsxCli, entry.scriptPath]
      : [entry.scriptPath];
  const csharpArgs = argParts
    .map((part) => `"${part.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`)
    .join(", ");

  const csharp = `using System;
using System.Diagnostics;
public static class RayvanSidecarLauncher {
  static readonly string[] Prefixed = new string[] { ${csharpArgs} };
  public static int Main(string[] args) {
    var psi = new ProcessStartInfo();
    psi.FileName = "node";
    psi.UseShellExecute = false;
    psi.Arguments = "";
    foreach (var part in Prefixed) {
      if (psi.Arguments.Length > 0) psi.Arguments += " ";
      psi.Arguments += Quote(part);
    }
    foreach (var arg in args) {
      psi.Arguments += " " + Quote(arg);
    }
    using (var process = Process.Start(psi)) {
      if (process == null) return 1;
      process.WaitForExit();
      return process.ExitCode;
    }
  }
  static string Quote(string value) {
    if (string.IsNullOrEmpty(value)) return "\\"\\"";
    if (value.IndexOfAny(new char[] {' ', '\\t', '"'}) < 0) return value;
    return "\\"" + value.Replace("\\"", "\\\\\\"") + "\\"";
  }
}
`;
  writeFileSync(csPath, csharp, "utf8");

  const ps = `
$ErrorActionPreference = 'Stop'
$code = Get-Content -Raw -LiteralPath '${csPath.replaceAll("'", "''")}'
Add-Type -TypeDefinition $code -Language CSharp -OutputAssembly '${exePath.replaceAll("'", "''")}' -OutputType ConsoleApplication
`;
  const result = spawnSync(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", ps],
    { encoding: "utf8" },
  );
  try {
    if (existsSync(csPath)) {
      // best-effort cleanup
      spawnSync("cmd.exe", ["/c", "del", "/f", "/q", csPath], {
        stdio: "ignore",
      });
    }
    // Remove empty tmp dir if possible
    spawnSync(
      "powershell.exe",
      [
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        `if (Test-Path -LiteralPath '${tmpDir.replaceAll("'", "''")}') { Remove-Item -LiteralPath '${tmpDir.replaceAll("'", "''")}' -Recurse -Force -ErrorAction SilentlyContinue }`,
      ],
      { stdio: "ignore" },
    );
  } catch {
    /* ignore */
  }

  if (result.status !== 0) {
    console.warn(
      `Windows .NET launcher compile failed for ${exePath}: ${
        result.stderr || result.stdout || `status ${result.status}`
      }`,
    );
    return false;
  }
  if (!existsSync(exePath) || statSync(exePath).size === 0) {
    console.warn(`Windows .NET launcher missing or empty: ${exePath}`);
    return false;
  }
  console.log(`Wrote .NET launcher ${exePath}`);
  return true;
}

function writeScaffoldNote(targetPath, entry, cmdPath) {
  writeFileSync(
    `${targetPath}.txt`,
    [
      "Sidecar scaffolding (not a production SEA/pkg binary).",
      `Launch: ${nodeInvocation(entry)}`,
      cmdPath ? `Dev launcher: ${cmdPath}` : "",
      "Replace the host-triple .exe/.bin with a real single-file executable before shipping installers.",
      "See apps/desktop/PACKAGING.md.",
    ]
      .filter(Boolean)
      .join("\n"),
    "utf8",
  );
}

function writeNodeWrapper(targetPath, entry) {
  if (process.platform === "win32") {
    const cmdPath = targetPath.endsWith(".exe")
      ? targetPath.replace(/\.exe$/i, ".cmd")
      : `${targetPath}.cmd`;
    writeWindowsCmd(cmdPath, entry);
    const compiled = tryWriteWindowsDotNetLauncher(targetPath, entry);
    if (!compiled) {
      writeFileSync(
        targetPath,
        Buffer.from(
          "Rayvan sidecar scaffolding placeholder. Use the sibling .cmd launcher or set RAYVAN_DAEMON_BIN. See PACKAGING.md.\n",
          "utf8",
        ),
      );
      console.log(
        `Wrote text placeholder ${targetPath} (use ${cmdPath} until SEA/pkg)`,
      );
    }
    writeScaffoldNote(targetPath, entry, cmdPath);
    return;
  }

  writeUnixLauncher(targetPath, entry);
  writeScaffoldNote(targetPath, entry);
}

function main() {
  mkdirSync(binariesDir, { recursive: true });
  const triple = hostTriple();
  const exe = process.platform === "win32" ? ".exe" : "";

  ensureBuilt("@rayvan/daemon");
  ensureBuilt("@rayvan/mcp-server");

  const daemonEntry = resolveEntry("daemon");
  const mcpEntry = resolveEntry("mcp");

  const rayvandTarget = join(binariesDir, `rayvand-${triple}${exe}`);
  const mcpTarget = join(binariesDir, `rayvan-mcp-${triple}${exe}`);

  if (isPlaceholderOrMissing(rayvandTarget)) {
    writeNodeWrapper(rayvandTarget, daemonEntry);
  } else {
    console.log(
      `Keeping existing ${rayvandTarget} (${statSync(rayvandTarget).size} bytes)`,
    );
  }
  if (isPlaceholderOrMissing(mcpTarget)) {
    writeNodeWrapper(mcpTarget, mcpEntry);
  } else {
    console.log(
      `Keeping existing ${mcpTarget} (${statSync(mcpTarget).size} bytes)`,
    );
  }

  if (daemonEntry.mode === "node") {
    copyFileSync(daemonEntry.scriptPath, join(binariesDir, "rayvand.mjs"));
  } else {
    writeFileSync(
      join(binariesDir, "rayvand.mjs"),
      `// Source fallback — run via tsx: ${daemonEntry.scriptPath}\n`,
      "utf8",
    );
  }
  if (mcpEntry.mode === "node") {
    copyFileSync(mcpEntry.scriptPath, join(binariesDir, "rayvan-mcp.mjs"));
  } else {
    writeFileSync(
      join(binariesDir, "rayvan-mcp.mjs"),
      `// Source fallback — run via tsx: ${mcpEntry.scriptPath}\n`,
      "utf8",
    );
  }

  const tauriConfPath = join(desktopRoot, "src-tauri", "tauri.conf.json");
  const tauriConf = JSON.parse(readFileSync(tauriConfPath, "utf8"));
  const externalBin = tauriConf?.bundle?.externalBin ?? [];
  for (const expected of ["binaries/rayvand", "binaries/rayvan-mcp"]) {
    if (!externalBin.includes(expected)) {
      throw new Error(
        `tauri.conf.json bundle.externalBin must include "${expected}"`,
      );
    }
  }

  console.log(`Sidecar scaffolding ready for ${triple}`);
  console.log(
    "Production tip: replace wrappers with Node SEA / pkg binaries named for the host triple.",
  );
}

main();
