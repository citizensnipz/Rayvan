import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { FINDING_SCHEMA_VERSION, findingId, projectId } from "@rayvan/core";
import { DaemonClient } from "@rayvan/daemon-client";
import { DaemonMethods, type DaemonEvent } from "@rayvan/daemon-contracts";
import { SqliteFindingRepository } from "@rayvan/local-database";
import { resetExampleLocalStore } from "@rayvan/plugin-example-local";
import { afterEach, describe, expect, it } from "vitest";

import { acquireDaemonLock, releaseDaemonLock } from "../src/lock.js";
import { DaemonRuntime } from "../src/runtime.js";
import { DaemonIpcServer } from "../src/server.js";

interface Harness {
  root: string;
  runtimeDir: string;
  endpoint: string;
  runtime: DaemonRuntime;
  server: DaemonIpcServer;
}

const harnesses: Harness[] = [];

afterEach(async () => {
  for (const harness of harnesses.splice(0)) {
    await harness.server.close();
    harness.runtime.close();
    releaseDaemonLock(harness.runtimeDir);
    rmSync(harness.root, { recursive: true, force: true });
  }
  resetExampleLocalStore();
});

describe("daemon IPC", () => {
  it("handshakes and correlates subsequent concurrent requests", async () => {
    const harness = await startHarness();
    const client = createClient(harness.endpoint, "test");

    const handshake = await client.connect();
    const [ping, status] = await Promise.all([
      client.call<{ ok: boolean }>(DaemonMethods.ping),
      client.call<{ endpoint: string; pluginHostStatus: string }>(
        DaemonMethods.status,
      ),
    ]);

    expect(handshake.sessionId).toMatch(/^sess_/);
    expect(ping.ok).toBe(true);
    expect(status.endpoint).toBe(harness.endpoint);
    expect(status.pluginHostStatus).toBe("ready");
    await client.close();
  });

  it("rejects MCP clients without registered credentials", async () => {
    const harness = await startHarness();
    const client = createClient(harness.endpoint, "mcp");

    await expect(client.connect()).rejects.toMatchObject({
      code: "UNAUTHENTICATED",
    });
    await client.close();
  });

  it("does not trust an unauthenticated client-type claim", async () => {
    const root = mkdtempSync(join(tmpdir(), "rayvand-auth-test-"));
    const runtimeDir = join(root, "run");
    const endpoint =
      process.platform === "win32"
        ? `\\\\.\\pipe\\rayvand-auth-test-${randomUUID()}`
        : join(runtimeDir, "rayvand.sock");
    const runtime = new DaemonRuntime({
      dataDir: join(root, "data"),
      runtimeDir,
      endpoint,
      provisionSystemClients: false,
    });
    const server = new DaemonIpcServer({ runtime });
    await server.start();
    harnesses.push({ root, runtimeDir, endpoint, runtime, server });
    const client = createClient(endpoint, "cli");

    await expect(client.connect()).rejects.toMatchObject({
      code: "UNAUTHENTICATED",
    });
    await client.close();
  });

  it("forwards events only after the client subscribes", async () => {
    const harness = await startHarness();
    const client = createClient(harness.endpoint, "test");
    const events: DaemonEvent[] = [];
    client.onEvent((event) => events.push(event));
    await client.connect();

    harness.runtime.events.emit({
      type: "daemon_status_changed",
      payload: { phase: "before" },
    });
    await delay(20);
    expect(events).toHaveLength(0);

    await client.subscribe();
    const received = new Promise<DaemonEvent>((resolve) => {
      const unsubscribe = client.onEvent((event) => {
        unsubscribe();
        resolve(event);
      });
    });
    harness.runtime.events.emit({
      type: "daemon_status_changed",
      payload: { phase: "after" },
    });

    await expect(received).resolves.toMatchObject({
      type: "daemon_status_changed",
      payload: { phase: "after" },
    });
    await client.close();
  });

  it("reuses a healthy lock and replaces a stale lock", async () => {
    const harness = await createHarness();
    const first = await acquireDaemonLock({
      runtimeDir: harness.runtimeDir,
      endpoint: harness.endpoint,
      probeAsTest: true,
    });
    expect(first.status).toBe("acquired");
    await harness.server.start();
    harnesses.push(harness);

    const reused = await acquireDaemonLock({
      runtimeDir: harness.runtimeDir,
      endpoint: harness.endpoint,
      probeAsTest: true,
    });
    expect(reused.status).toBe("reused");

    await harness.server.close();
    harness.runtime.close();
    harnesses.splice(harnesses.indexOf(harness), 1);
    releaseDaemonLock(harness.runtimeDir);

    writeFileSync(
      join(harness.runtimeDir, "rayvand.lock"),
      JSON.stringify({
        pid: 2_147_483_647,
        protocolVersion: "1",
        endpoint: harness.endpoint,
        startedAt: "1970-01-01T00:00:00.000Z",
      }),
    );
    const recovered = await acquireDaemonLock({
      runtimeDir: harness.runtimeDir,
      endpoint: harness.endpoint,
      probeAsTest: true,
    });
    expect(recovered.status).toBe("acquired");

    releaseDaemonLock(harness.runtimeDir);
    rmSync(harness.root, { recursive: true, force: true });
  });

  it("closes clients and removes the Unix socket gracefully", async () => {
    const harness = await startHarness();
    const client = createClient(harness.endpoint, "test");
    await client.connect();

    await harness.server.close();
    await expect(client.call(DaemonMethods.ping)).rejects.toThrow(
      /Not connected|closed|EPIPE/,
    );
    if (process.platform !== "win32") {
      expect(existsSync(harness.endpoint)).toBe(false);
    }
    await client.close();
  });

  it("lists example-local as available via the in-process host", async () => {
    const harness = await startHarness();
    const client = createClient(harness.endpoint, "test");
    await client.connect();

    const plugins = await client.call<
      Array<{ pluginId: string; status: string; host: string }>
    >(DaemonMethods.listPlugins);

    expect(plugins).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          pluginId: "example-local",
          status: "available",
          host: "in_process",
        }),
      ]),
    );
    await client.close();
  });

  it("syncs, generates, approves, applies, and verifies via example-local", async () => {
    const harness = await startHarness();
    const client = createClient(harness.endpoint, "test");
    await client.connect();

    const project = await client.call<{ id: string }>(DaemonMethods.createProject, {
      name: "Plugin Path",
    });

    const sync = await client.call<{
      connectionId: string;
      bindings: Array<{ id: string }>;
      discovered: unknown[];
    }>(DaemonMethods.syncProject, { projectId: project.id });

    expect(sync.connectionId).toBeTruthy();
    expect(sync.discovered.length).toBeGreaterThan(0);
    expect(sync.bindings.length).toBeGreaterThan(0);

    const bindingId = sync.bindings[0]!.id;
    const inspected = await client.call<{
      observed: { attributes: { port: number } };
    }>(DaemonMethods.inspectResource, {
      projectId: project.id,
      resourceId: bindingId,
    });
    expect(inspected.observed.attributes.port).toBe(3000);

    const generated = await client.call<{
      plan: { id: string; plan: { operations: Array<{ id: string }> } };
    }>(DaemonMethods.generateChangePlan, {
      projectId: project.id,
      resourceBindingId: bindingId,
      desiredAttributes: { port: 3010 },
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
    await client.close();
  });

  it("reopens a dismissed finding", async () => {
    const harness = await startHarness();
    const client = createClient(harness.endpoint, "test");
    await client.connect();

    const project = await client.call<{ id: string }>(DaemonMethods.createProject, {
      name: "Findings",
    });
    const findings = new SqliteFindingRepository(harness.runtime.db);
    const now = new Date().toISOString();
    const id = findingId(`finding_${randomUUID()}`);
    await findings.save({
      id,
      projectId: projectId(project.id),
      ruleId: "test.rule",
      source: { type: "rayvan" },
      category: "configuration",
      severity: "warning",
      title: "Test finding",
      summary: "Needs reopen",
      status: "open",
      fingerprint: `fp_${randomUUID()}`,
      fingerprintVersion: "1",
      evidence: [{ type: "message", message: "fixture" }],
      firstDetectedAt: now,
      lastDetectedAt: now,
      occurrenceCount: 1,
      metadata: {},
      schemaVersion: FINDING_SCHEMA_VERSION,
    });

    await client.call(DaemonMethods.dismissFinding, {
      findingId: id,
      reason: "temporary",
    });

    const reopened = await client.call<{ status: string }>(
      DaemonMethods.reopenFinding,
      { findingId: id },
    );
    expect(reopened.status).toBe("open");
    await client.close();
  });

  it("adopts and ignores discovered configuration occurrences", async () => {
    const harness = await startHarness();
    const client = createClient(harness.endpoint, "test");
    await client.connect();

    const project = await client.call<{ id: string }>(DaemonMethods.createProject, {
      name: "Config Adopt",
    });
    const environment = await client.call<{ id: string }>(
      DaemonMethods.createEnvironment,
      {
        projectId: project.id,
        name: "local",
        kind: "local",
      },
    );

    const key = await harness.runtime.configurationService.upsertKeyByName(
      project.id,
      "DEBUG_MODE",
      { source: "discovered", valueType: "boolean" },
    );
    const occurrence = await harness.runtime.configurationService.upsertOccurrence({
      configurationKeyId: key.id,
      projectId: project.id,
      environmentId: environment.id,
      pluginId: "example-local",
      connectionId: "conn_test",
      discoveredResourceId: "res_test",
      providerKey: "DEBUG_MODE",
      valueAccess: "readable",
      observedValue: "false",
    });

    const adopted = await client.call<{
      key: { source: string };
      occurrence: { id: string };
    }>(DaemonMethods.adoptDiscoveredConfiguration, {
      projectId: project.id,
      occurrenceId: occurrence.id,
      environmentId: environment.id,
    });
    expect(adopted.key.source).toBe("manual");

    const otherKey = await harness.runtime.configurationService.upsertKeyByName(
      project.id,
      "OTHER",
      { source: "discovered", valueType: "string" },
    );
    const ignoredOccurrence =
      await harness.runtime.configurationService.upsertOccurrence({
        configurationKeyId: otherKey.id,
        projectId: project.id,
        environmentId: environment.id,
        pluginId: "example-local",
        connectionId: "conn_ignore",
        discoveredResourceId: "res_ignore",
        providerKey: "OTHER",
        valueAccess: "readable",
        observedValue: "x",
      });

    await client.call(DaemonMethods.ignoreDiscoveredConfiguration, {
      projectId: project.id,
      occurrenceId: ignoredOccurrence.id,
    });

    const unmanaged = await client.call<Array<{ id: string }>>(
      DaemonMethods.listUnmanagedConfiguration,
      { projectId: project.id },
    );
    expect(unmanaged.map((item) => item.id)).not.toContain(ignoredOccurrence.id);
    await client.close();
  });

  it("rejects blind retry of interrupted applies and allows clean failed retry", async () => {
    const harness = await startHarness();
    const client = createClient(harness.endpoint, "test");
    await client.connect();

    const project = await client.call<{ id: string }>(DaemonMethods.createProject, {
      name: "Retry",
    });
    const sync = await client.call<{ bindings: Array<{ id: string }> }>(
      DaemonMethods.syncProject,
      { projectId: project.id },
    );
    const bindingId = sync.bindings[0]!.id;

    const generated = await client.call<{
      plan: { id: string; plan: { operations: Array<{ id: string }> } };
    }>(DaemonMethods.generateChangePlan, {
      projectId: project.id,
      resourceBindingId: bindingId,
      desiredAttributes: { port: 3020 },
    });
    await client.call(DaemonMethods.approveChangePlan, {
      changePlanId: generated.plan.id,
    });

    // Simulate interrupted apply: status applying, no completed apply.
    await harness.runtime.pluginRepos.changePlans.setStatus(
      generated.plan.id,
      "applying",
    );
    await expect(
      client.call(DaemonMethods.retryFailedChange, {
        changePlanId: generated.plan.id,
      }),
    ).rejects.toMatchObject({ code: "VALIDATION_FAILED" });

    // Clean failed apply path.
    await harness.runtime.pluginRepos.changePlans.setStatus(
      generated.plan.id,
      "approved",
    );
    await harness.runtime.changeApprovalService.beginApply(generated.plan.id);
    await harness.runtime.changeApprovalService.completeApply({
      changePlanId: generated.plan.id,
      executionId: "exec_fail",
      status: "failed",
      startedAt: new Date().toISOString(),
      error: {
        code: "execution_failed",
        message: "provider rejected",
        pluginId: "example-local",
        capability: "apply",
        retryable: true,
      },
    });

    const retried = await client.call<{ status: string }>(
      DaemonMethods.retryFailedChange,
      { changePlanId: generated.plan.id },
    );
    expect(retried.status).toBe("succeeded");
    await client.close();
  });
});

async function startHarness(): Promise<Harness> {
  const harness = await createHarness();
  await harness.server.start();
  harnesses.push(harness);
  return harness;
}

async function createHarness(): Promise<Harness> {
  resetExampleLocalStore();
  const root = mkdtempSync(join(tmpdir(), "rayvand-test-"));
  const runtimeDir = join(root, "run");
  const endpoint =
    process.platform === "win32"
      ? `\\\\.\\pipe\\rayvand-test-${randomUUID()}`
      : join(runtimeDir, "rayvand.sock");
  const runtime = new DaemonRuntime({
    dataDir: join(root, "data"),
    runtimeDir,
    endpoint,
    provisionSystemClients: false,
    allowUnauthenticatedTestClient: true,
  });
  const server = new DaemonIpcServer({ runtime });
  return { root, runtimeDir, endpoint, runtime, server };
}

function createClient(
  endpoint: string,
  clientType: "test" | "mcp" | "cli",
): DaemonClient {
  return new DaemonClient({
    endpoint,
    clientType,
    clientVersion: "test",
    connectTimeoutMs: 1_000,
  });
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
