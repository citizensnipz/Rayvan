export type DaemonLifecycleState =
  "starting" | "migrating" | "ready" | "degraded" | "shutting_down" | "stopped";

export interface DaemonStatus {
  state: DaemonLifecycleState;
  version: string;
  protocolVersion: string;
  pid: number;
  uptimeMs: number;
  databasePath: string;
  databaseSchemaVersion: number;
  endpoint: string;
  connectedClients: number;
  activeOperations: number;
  pendingApprovals: number;
  pluginHostStatus: "ready" | "degraded" | "unavailable";
}

export interface DaemonDiagnostics {
  status: DaemonStatus;
  lockPath: string;
  runtimeDir: string;
  dataDir: string;
  logPath?: string;
  recentErrors: Array<{ timestamp: string; code: string; message: string }>;
  redacted: true;
}
