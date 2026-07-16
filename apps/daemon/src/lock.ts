import {
  existsSync,
  mkdirSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";

import {
  DaemonClient,
  daemonCredentialStorePath,
  daemonEndpointPath,
  daemonLockPath,
  daemonPidPath,
  defaultRayvanRuntimeDir,
  LocalClientCredentialStore,
} from "@rayvan/daemon-client";
import {
  BUILT_IN_LOCAL_CLIENT_IDS,
  DAEMON_PROTOCOL_VERSION,
} from "@rayvan/daemon-contracts";

export interface DaemonLockInfo {
  pid: number;
  protocolVersion: string;
  endpoint: string;
  startedAt: string;
  dataDir?: string;
}

export type AcquireLockResult =
  | { status: "acquired"; lockPath: string }
  | { status: "reused"; info: DaemonLockInfo }
  | { status: "incompatible"; info: DaemonLockInfo; reason: string };

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function probeDaemon(
  endpoint: string,
  options: { dataDir?: string; asTest?: boolean },
): Promise<boolean> {
  const credentialStore = options.dataDir
    ? new LocalClientCredentialStore(daemonCredentialStorePath(options.dataDir))
    : undefined;
  const client = new DaemonClient({
    endpoint,
    clientType: options.asTest ? "test" : "cli",
    clientVersion: "0.0.1",
    clientId: options.asTest ? undefined : BUILT_IN_LOCAL_CLIENT_IDS.cli,
    clientCredential: options.asTest
      ? undefined
      : (credentialStore?.resolve(BUILT_IN_LOCAL_CLIENT_IDS.cli) ?? undefined),
  });
  try {
    await client.connect();
    await client.call("system.ping", {});
    await client.close();
    return true;
  } catch {
    await client.close().catch(() => undefined);
    return false;
  }
}

export async function acquireDaemonLock(options?: {
  runtimeDir?: string;
  endpoint?: string;
  dataDir?: string;
  probeAsTest?: boolean;
  force?: boolean;
  attempt?: number;
}): Promise<AcquireLockResult> {
  const runtimeDir = options?.runtimeDir ?? defaultRayvanRuntimeDir();
  const lockPath = daemonLockPath(runtimeDir);
  const pidPath = daemonPidPath(runtimeDir);
  const endpoint = options?.endpoint ?? daemonEndpointPath(runtimeDir);

  mkdirSync(runtimeDir, { recursive: true, mode: 0o700 });

  if (existsSync(lockPath)) {
    let info: DaemonLockInfo | null = null;
    try {
      info = JSON.parse(readFileSync(lockPath, "utf8")) as DaemonLockInfo;
    } catch {
      info = null;
    }

    if (info) {
      const alive = isProcessAlive(info.pid);
      const healthy = alive
        ? await probeDaemon(info.endpoint || endpoint, {
            dataDir: info.dataDir ?? options?.dataDir,
            asTest: options?.probeAsTest,
          })
        : false;

      if (healthy) {
        if (info.protocolVersion !== DAEMON_PROTOCOL_VERSION) {
          return {
            status: "incompatible",
            info,
            reason: `Existing daemon protocol ${info.protocolVersion} != ${DAEMON_PROTOCOL_VERSION}`,
          };
        }
        return { status: "reused", info };
      }

      if (alive && !healthy) {
        return {
          status: "incompatible",
          info,
          reason: "Process exists but daemon is not healthy",
        };
      }

      // Stale lock recovery
      try {
        unlinkSync(lockPath);
      } catch {
        /* ignore */
      }
      try {
        unlinkSync(pidPath);
      } catch {
        /* ignore */
      }
    } else {
      try {
        unlinkSync(lockPath);
      } catch {
        /* ignore */
      }
      try {
        unlinkSync(pidPath);
      } catch {
        /* ignore */
      }
    }
  }

  const info: DaemonLockInfo = {
    pid: process.pid,
    protocolVersion: DAEMON_PROTOCOL_VERSION,
    endpoint,
    startedAt: new Date().toISOString(),
    dataDir: options?.dataDir,
  };
  try {
    writeFileSync(lockPath, JSON.stringify(info, null, 2), {
      mode: 0o600,
      flag: "wx",
    });
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "EEXIST" && (options?.attempt ?? 0) < 1) {
      return acquireDaemonLock({
        ...options,
        attempt: (options?.attempt ?? 0) + 1,
      });
    }
    throw error;
  }
  writeFileSync(pidPath, String(process.pid), { mode: 0o600 });
  return { status: "acquired", lockPath };
}

export function releaseDaemonLock(runtimeDir = defaultRayvanRuntimeDir()): void {
  const lockPath = daemonLockPath(runtimeDir);
  const pidPath = daemonPidPath(runtimeDir);
  if (existsSync(lockPath)) {
    try {
      const info = JSON.parse(readFileSync(lockPath, "utf8")) as DaemonLockInfo;
      if (info.pid !== process.pid) {
        return;
      }
    } catch {
      return;
    }
  }
  for (const path of [lockPath, pidPath]) {
    if (existsSync(path)) {
      try {
        unlinkSync(path);
      } catch {
        /* ignore */
      }
    }
  }
  // Best-effort socket cleanup on unix
  const endpoint = daemonEndpointPath(runtimeDir);
  if (!endpoint.startsWith("\\\\.\\pipe\\") && existsSync(endpoint)) {
    try {
      unlinkSync(endpoint);
    } catch {
      /* ignore */
    }
  }
  try {
    mkdirSync(dirname(lockPath), { recursive: true });
  } catch {
    /* ignore */
  }
}
