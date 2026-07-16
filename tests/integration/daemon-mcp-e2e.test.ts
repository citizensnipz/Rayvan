/**
 * Real multi-process end-to-end scenario for local daemon + MCP.
 *
 * Spawns an isolated `rayvand serve` child process (temp data/runtime/endpoint),
 * exercises example-local sync → plan → approve → apply → verify over IPC, and
 * optionally calls one MCP read tool against the same daemon.
 *
 * Run:
 *   pnpm test:integration
 *   pnpm exec vitest run --config vitest.config.ts tests/integration/daemon-mcp-e2e.test.ts
 *
 * Requires Node 20+. Builds daemon dist on demand when missing.
 */
import { spawn, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { FINDING_SCHEMA_VERSION, findingId, projectId } from "@rayvan/core";
import { DaemonClient } from "@rayvan/daemon-client";
import { DaemonMethods } from "@rayvan/daemon-contracts";
import { SqliteFindingRepository } from "@rayvan/local-database";
import { LocalDatabaseConnection } from "@rayvan/local-database/sqlite";
import { resetExampleLocalStore } from "@rayvan/plugin-example-local";
import { afterEach, describe, expect, it } from "vitest";

import { createRayvanMcpServer } from "../../apps/mcp-server/src/server/index.ts";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const DAEMON_DIST = join(repoRoot, "apps/daemon/dist/main.js");
const DAEMON_SRC = join(repoRoot, "apps/daemon/src/main.ts");

interface E2EHarness {
  root: string;
  dataDir: string;
  runtimeDir: string;
  endpoint: string;
  child: ChildProcess;
  client: DaemonClient;
}

const harnesses: E2EHarness[] = [];

afterEach(async () => {
  for (const harness of harnesses.splice(0)) {
    await tearDown(harness);
  }
  resetExampleLocalStore();
});

describe("daemon MCP real-process E2E", () => {
  it(
    "syncs, plans, applies, verifies example-local across a real rayvand process",
    async () => {
      const harness = await startRealDaemon();
      const { client, dataDir } = harness;

      const project = await client.call<{ id: string }>(
        DaemonMethods.createProject,
        { name: `E2E ${randomUUID().slice(0, 8)}` },
      );
      const environment = await client.call<{ id: string }>(
        DaemonMethods.createEnvironment,
        {
          projectId: project.id,
          name: "local",
          kind: "local",
        },
      );
      expect(environment.id).toBeTruthy();

      const sync = await client.call<{
        connectionId: string;
        bindings: Array<{ id: string }>;
        discovered: unknown[];
      }>(DaemonMethods.syncProject, {
        projectId: project.id,
        environmentId: environment.id,
      });
      expect(sync.connectionId).toBeTruthy();
      expect(sync.discovered.length).toBeGreaterThan(0);
      expect(sync.bindings.length).toBeGreaterThan(0);
      const bindingId = sync.bindings[0]!.id;

      const finding = await seedDebugModeFinding({
        dataDir,
        projectId: project.id,
        environmentId: environment.id,
        resourceBindingId: bindingId,
      });

      const listed = await client.call<Array<{ id: string; title: string }>>(
        DaemonMethods.listFindings,
        { projectId: project.id },
      );
      expect(listed.map((f) => f.id)).toContain(finding.id);
      expect(listed.find((f) => f.id === finding.id)?.title).toMatch(
        /DEBUG_MODE/,
      );

      const generated = await client.call<{
        plan: { id: string; plan: { operations: Array<{ id: string }> } };
      }>(DaemonMethods.generatePlanFromFinding, {
        findingId: finding.id,
      });
      expect(generated.plan.plan.operations.length).toBeGreaterThan(0);

      await client.call(DaemonMethods.approveChangePlan, {
        changePlanId: generated.plan.id,
      });

      const applied = await client.call<{ status: string }>(
        DaemonMethods.applyChangePlan,
        { changePlanId: generated.plan.id },
      );
      expect(applied.status).toBe("succeeded");

      const verified = await client.call<{ status: string }>(
        DaemonMethods.verifyChangePlan,
        { changePlanId: generated.plan.id },
      );
      expect(verified.status).toBe("verified");

      const after = await client.call<{
        observed: { attributes: { port: number } };
      }>(DaemonMethods.inspectResource, {
        projectId: project.id,
        resourceId: bindingId,
      });
      expect(after.observed.attributes.port).toBe(3010);

      const ops = await client.call<
        Array<{ type: string; status: string }>
      >(DaemonMethods.listOperations, { projectId: project.id });
      expect(ops.some((op) => op.type === "change_apply" && op.status === "succeeded")).toBe(
        true,
      );

      const resolved = await client.call<{ status: string } | null>(
        DaemonMethods.getFinding,
        { findingId: finding.id },
      );
      expect(resolved?.status).toBe("resolved");

      // Optional MCP read tool against the same daemon session.
      const mcpServer = createRayvanMcpServer(client);
      const mcpClient = new Client({ name: "e2e", version: "0.0.1" });
      const [clientTransport, serverTransport] =
        InMemoryTransport.createLinkedPair();
      await mcpServer.connect(serverTransport);
      await mcpClient.connect(clientTransport);
      try {
        const toolResult = await mcpClient.callTool({
          name: "list_projects",
          arguments: {},
        });
        expect(toolResult.isError).not.toBe(true);
        const structured = toolResult.structuredContent as {
          data?: Array<{ id: string }>;
        };
        expect(structured.data?.some((p) => p.id === project.id)).toBe(true);
      } finally {
        await mcpClient.close();
        await mcpServer.close();
      }
    },
    90_000,
  );
});

async function startRealDaemon(): Promise<E2EHarness> {
  resetExampleLocalStore();
  ensureDaemonEntry();

  const root = mkdtempSync(join(tmpdir(), "rayvan-e2e-"));
  const dataDir = join(root, "data");
  const runtimeDir = join(root, "run");
  const endpoint =
    process.platform === "win32"
      ? `\\\\.\\pipe\\rayvan-e2e-${randomUUID()}`
      : join(runtimeDir, "rayvand.sock");

  const { command, args } = daemonSpawnCommand([
    "serve",
    "--data-dir",
    dataDir,
    "--runtime-dir",
    runtimeDir,
    "--endpoint",
    endpoint,
  ]);

  const child = spawn(command, args, {
    cwd: repoRoot,
    env: {
      ...process.env,
      RAYVAN_DATA_DIR: dataDir,
      RAYVAN_RUNTIME_DIR: runtimeDir,
      RAYVAN_DAEMON_ENDPOINT: endpoint,
      RAYVAN_ALLOW_UNAUTHENTICATED_TEST_CLIENT: "1",
    },
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  let stderr = "";
  child.stderr?.on("data", (chunk: Buffer) => {
    stderr += chunk.toString("utf8");
  });
  child.stdout?.on("data", () => {
    /* absorb serve noise */
  });

  const client = new DaemonClient({
    endpoint,
    clientType: "test",
    clientVersion: "e2e",
    connectTimeoutMs: 2_000,
  });

  const deadline = Date.now() + 30_000;
  let lastError: unknown;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(
        `rayvand exited early (code=${String(child.exitCode)}): ${stderr || "(no stderr)"}`,
      );
    }
    try {
      await client.connect();
      const harness: E2EHarness = {
        root,
        dataDir,
        runtimeDir,
        endpoint,
        child,
        client,
      };
      harnesses.push(harness);
      return harness;
    } catch (error) {
      lastError = error;
      await client.close().catch(() => undefined);
      await sleep(150);
    }
  }

  child.kill();
  throw new Error(
    `Timed out waiting for rayvand at ${endpoint}: ${
      lastError instanceof Error ? lastError.message : String(lastError)
    }\n${stderr}`,
  );
}

async function tearDown(harness: E2EHarness): Promise<void> {
  try {
    await harness.client.call(DaemonMethods.shutdown).catch(() => undefined);
  } catch {
    /* ignore */
  }
  await harness.client.close().catch(() => undefined);

  await waitForExit(harness.child, 5_000);
  if (harness.child.exitCode === null && harness.child.pid) {
    try {
      harness.child.kill("SIGKILL");
    } catch {
      /* ignore */
    }
    await waitForExit(harness.child, 2_000);
  }

  rmSync(harness.root, { recursive: true, force: true });
}

async function seedDebugModeFinding(input: {
  dataDir: string;
  projectId: string;
  environmentId: string;
  resourceBindingId: string;
}): Promise<{ id: string }> {
  const dbPath = join(input.dataDir, "rayvan.db");
  const db = new LocalDatabaseConnection(dbPath);
  try {
    const findings = new SqliteFindingRepository(db);
    const now = new Date().toISOString();
    const id = findingId(`finding_${randomUUID()}`);
    await findings.save({
      id,
      projectId: projectId(input.projectId),
      ruleId: "example-local.debug-mode",
      source: { type: "plugin", pluginId: "example-local" },
      category: "configuration",
      severity: "warning",
      title: "DEBUG_MODE enabled in local environment",
      summary: "DEBUG_MODE should be false for this scenario",
      status: "open",
      fingerprint: `fp_debug_${randomUUID()}`,
      fingerprintVersion: "1",
      environmentId: input.environmentId as never,
      resourceBindingId: input.resourceBindingId,
      evidence: [
        {
          type: "message",
          message: "E2E fixture: DEBUG_MODE true",
        },
      ],
      firstDetectedAt: now,
      lastDetectedAt: now,
      occurrenceCount: 1,
      metadata: {},
      schemaVersion: FINDING_SCHEMA_VERSION,
    });
    return { id };
  } finally {
    db.close();
  }
}

function ensureDaemonEntry(): void {
  if (existsSync(DAEMON_DIST) || existsSync(DAEMON_SRC)) {
    return;
  }
  throw new Error(
    `Daemon entry missing. Expected ${DAEMON_DIST} or ${DAEMON_SRC}`,
  );
}

function daemonSpawnCommand(serveArgs: string[]): {
  command: string;
  args: string[];
} {
  // Prefer tsx + source so E2E picks up unbuilt daemon changes (Windows-safe).
  // Set RAYVAN_E2E_USE_DIST=1 to force the compiled entry when present.
  if (process.env.RAYVAN_E2E_USE_DIST === "1" && existsSync(DAEMON_DIST)) {
    return { command: process.execPath, args: [DAEMON_DIST, ...serveArgs] };
  }

  const tsxCli = resolveTsxCli();
  return {
    command: process.execPath,
    args: [
      tsxCli,
      "--tsconfig",
      join(repoRoot, "apps/daemon/tests/tsconfig.typecheck.json"),
      DAEMON_SRC,
      ...serveArgs,
    ],
  };
}

function resolveTsxCli(): string {
  const candidates = [
    join(repoRoot, "apps/daemon/node_modules/tsx/dist/cli.mjs"),
    join(repoRoot, "node_modules/tsx/dist/cli.mjs"),
    join(repoRoot, "apps/daemon/node_modules/tsx/dist/cli.js"),
    join(repoRoot, "node_modules/tsx/dist/cli.js"),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error(
    "tsx CLI not found. Install workspace deps or set RAYVAN_E2E_USE_DIST=1 with a built daemon dist.",
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForExit(
  child: ChildProcess,
  timeoutMs: number,
): Promise<void> {
  if (child.exitCode !== null) {
    return;
  }
  await Promise.race([
    new Promise<void>((resolve) => child.once("exit", () => resolve())),
    sleep(timeoutMs),
  ]);
}
