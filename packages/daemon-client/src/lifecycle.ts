import { spawn, type ChildProcess } from "node:child_process";

import { DaemonClient } from "./client.js";
import { daemonEndpointPath } from "./paths.js";
import type { DaemonClientTransportOptions } from "./transport.js";

export interface LaunchOrAttachOptions extends Omit<
  DaemonClientTransportOptions,
  "clientType" | "clientVersion"
> {
  clientType?: DaemonClientTransportOptions["clientType"];
  clientVersion?: string;
  /** Absolute path or command name for rayvand. */
  daemonBinary?: string;
  spawnArgs?: string[];
  maxWaitMs?: number;
}

export interface LaunchOrAttachResult {
  client: DaemonClient;
  spawned: boolean;
  child?: ChildProcess;
}

/**
 * Attach to a healthy daemon, or spawn one and wait for handshake.
 */
export async function launchOrAttachDaemon(
  options: LaunchOrAttachOptions = {},
): Promise<LaunchOrAttachResult> {
  const clientType = options.clientType ?? "cli";
  const endpoint = options.endpoint ?? daemonEndpointPath();

  const attach = async (): Promise<DaemonClient | null> => {
    const client = new DaemonClient({
      ...options,
      clientType,
      clientVersion: options.clientVersion ?? "0.0.1",
      endpoint,
    });
    try {
      await client.connect();
      return client;
    } catch (error) {
      await client.close().catch(() => undefined);
      if (isDaemonUnavailable(error)) {
        return null;
      }
      throw error;
    }
  };

  const existing = await attach();
  if (existing) {
    return { client: existing, spawned: false };
  }

  const binary = resolveDaemonBinary(options.daemonBinary);
  const child = spawn(binary, options.spawnArgs ?? ["serve"], {
    detached: true,
    stdio: "ignore",
    env: {
      ...process.env,
      RAYVAN_DAEMON_ENDPOINT: endpoint,
      ...(options.endpoint ? {} : {}),
    },
    windowsHide: true,
  });
  let spawnError: Error | undefined;
  let childExit: { code: number | null; signal: NodeJS.Signals | null } | undefined;
  child.once("error", (error) => {
    spawnError = error;
  });
  child.once("exit", (code, signal) => {
    childExit = { code, signal };
  });
  child.unref();

  const deadline = Date.now() + (options.maxWaitMs ?? 15_000);
  while (Date.now() < deadline) {
    if (spawnError) {
      throw new Error(`Failed to launch rayvand using ${binary}`, {
        cause: spawnError,
      });
    }
    if (childExit) {
      throw new Error(
        `rayvand exited before becoming healthy (code=${String(childExit.code)}, signal=${String(childExit.signal)})`,
      );
    }
    const client = await attach();
    if (client) {
      return { client, spawned: true, child };
    }
    await sleep(150);
  }

  throw new Error(
    `Failed to start or attach to rayvand at ${endpoint} using ${binary}`,
  );
}

function resolveDaemonBinary(explicit?: string): string {
  if (explicit) {
    return explicit;
  }
  if (process.env.RAYVAN_DAEMON_BIN) {
    return process.env.RAYVAN_DAEMON_BIN;
  }
  return process.platform === "win32" ? "rayvand.exe" : "rayvand";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isDaemonUnavailable(error: unknown): boolean {
  const value = error as Error & { code?: string; cause?: { code?: string } };
  const code = value.code ?? value.cause?.code;
  return (
    code === "ENOENT" ||
    code === "ECONNREFUSED" ||
    code === "EPIPE" ||
    code === "ETIMEDOUT" ||
    /timed out connecting to daemon/i.test(value.message ?? "")
  );
}
