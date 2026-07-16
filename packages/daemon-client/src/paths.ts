import { createHash } from "node:crypto";
import { homedir, tmpdir, userInfo } from "node:os";
import { join } from "node:path";

function userScopeId(): string {
  try {
    const info = userInfo();
    const raw = `${info.uid}:${info.username}`;
    return createHash("sha256").update(raw).digest("hex").slice(0, 12);
  } catch {
    return createHash("sha256").update(homedir()).digest("hex").slice(0, 12);
  }
}

export function defaultRayvanDataDir(): string {
  if (process.env.RAYVAN_DATA_DIR) {
    return process.env.RAYVAN_DATA_DIR;
  }
  if (process.platform === "win32") {
    const base = process.env.APPDATA ?? join(homedir(), "AppData", "Roaming");
    return join(base, "com.rayvan.desktop");
  }
  if (process.platform === "darwin") {
    return join(homedir(), "Library", "Application Support", "com.rayvan.desktop");
  }
  const xdg = process.env.XDG_DATA_HOME ?? join(homedir(), ".local", "share");
  return join(xdg, "rayvan");
}

export function defaultRayvanRuntimeDir(): string {
  if (process.env.RAYVAN_RUNTIME_DIR) {
    return process.env.RAYVAN_RUNTIME_DIR;
  }
  if (process.platform === "win32") {
    const base = process.env.LOCALAPPDATA ?? join(homedir(), "AppData", "Local");
    return join(base, "rayvan", "run");
  }
  if (process.platform === "darwin") {
    return join(homedir(), "Library", "Caches", "rayvan", "run");
  }
  const xdg = process.env.XDG_RUNTIME_DIR ?? join(tmpdir(), `rayvan-${userScopeId()}`);
  return join(xdg, "rayvan");
}

export function daemonEndpointPath(runtimeDir = defaultRayvanRuntimeDir()): string {
  if (process.env.RAYVAN_DAEMON_ENDPOINT) {
    return process.env.RAYVAN_DAEMON_ENDPOINT;
  }
  const scope = userScopeId();
  if (process.platform === "win32") {
    return `\\\\.\\pipe\\rayvan-${scope}`;
  }
  return join(runtimeDir, "rayvand.sock");
}

export function daemonLockPath(runtimeDir = defaultRayvanRuntimeDir()): string {
  return join(runtimeDir, "rayvand.lock");
}

export function daemonPidPath(runtimeDir = defaultRayvanRuntimeDir()): string {
  return join(runtimeDir, "rayvand.pid");
}

export function daemonCredentialStorePath(dataDir = defaultRayvanDataDir()): string {
  return join(dataDir, "credentials", "local-clients.json");
}

export function databasePath(dataDir = defaultRayvanDataDir()): string {
  return join(dataDir, "rayvan.db");
}
