import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { DaemonClient } from "@rayvan/daemon-client";
import { DaemonMethods, type DaemonEvent } from "@rayvan/daemon-contracts";
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
});

describe("daemon IPC", () => {
  it("handshakes and correlates subsequent concurrent requests", async () => {
    const harness = await startHarness();
    const client = createClient(harness.endpoint, "test");

    const handshake = await client.connect();
    const [ping, status] = await Promise.all([
      client.call<{ ok: boolean }>(DaemonMethods.ping),
      client.call<{ endpoint: string }>(DaemonMethods.status),
    ]);

    expect(handshake.sessionId).toMatch(/^sess_/);
    expect(ping.ok).toBe(true);
    expect(status.endpoint).toBe(harness.endpoint);
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
});

async function startHarness(): Promise<Harness> {
  const harness = await createHarness();
  await harness.server.start();
  harnesses.push(harness);
  return harness;
}

async function createHarness(): Promise<Harness> {
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
