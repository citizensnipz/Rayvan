import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  DaemonMethods,
  type ApprovalRequestRecord,
  type DaemonDiagnostics,
  type DaemonEvent,
  type DaemonStatus,
  type LocalClientPublicView,
  type OperationRecord,
} from "@rayvan/daemon-contracts";

export interface DaemonStatusSnapshot {
  connected: boolean;
  endpoint: string;
  spawned: boolean;
  sessionId?: string;
  daemonVersion?: string;
  authenticatedClientId?: string;
  lastError?: string;
  status?: DaemonStatus;
}

export interface DaemonCommandError {
  code: string;
  message: string;
  id?: string;
  data?: unknown;
}

export class DaemonClientError extends Error {
  readonly code: string;
  readonly data?: unknown;

  constructor(error: DaemonCommandError) {
    super(error.message);
    this.name = "DaemonClientError";
    this.code = error.code;
    this.data = error.data;
  }
}

function isCommandError(error: unknown): error is DaemonCommandError {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    "message" in error &&
    typeof (error as DaemonCommandError).code === "string" &&
    typeof (error as DaemonCommandError).message === "string"
  );
}

async function invokeDaemon<T>(
  command: string,
  payload?: Record<string, unknown>,
): Promise<T> {
  try {
    return await invoke<T>(command, payload);
  } catch (error) {
    if (isCommandError(error)) {
      throw new DaemonClientError(error);
    }
    throw error;
  }
}

export async function getDaemonStatus(): Promise<DaemonStatusSnapshot> {
  return invokeDaemon<DaemonStatusSnapshot>("daemon_status");
}

export async function reconnectDaemon(): Promise<DaemonStatusSnapshot> {
  return invokeDaemon<DaemonStatusSnapshot>("daemon_reconnect");
}

export async function daemonRequest<T = unknown>(
  method: string,
  params?: unknown,
): Promise<T> {
  return invokeDaemon<T>("daemon_request", {
    method,
    params: params ?? {},
  });
}

export function listenDaemonEvents(
  listener: (event: DaemonEvent) => void,
): Promise<UnlistenFn> {
  return listen<DaemonEvent>("daemon://event", (event) => {
    listener(event.payload);
  });
}

/** Thin typed helpers over `daemon_request`. */
export const desktopDaemon = {
  status: () => daemonRequest<DaemonStatus>(DaemonMethods.status),
  diagnostics: () => daemonRequest<DaemonDiagnostics>(DaemonMethods.diagnostics),
  listProjects: (params?: { includeArchived?: boolean }) =>
    daemonRequest(DaemonMethods.listProjects, params ?? {}),
  getProject: (projectId: string) =>
    daemonRequest(DaemonMethods.getProject, { projectId }),
  createProject: (params: { name: string; description?: string }) =>
    daemonRequest(DaemonMethods.createProject, params),
  updateProject: (params: {
    projectId: string;
    name?: string;
    description?: string;
    archived?: boolean;
  }) => daemonRequest(DaemonMethods.updateProject, params),
  listEnvironments: (
    projectId: string,
    options?: { includeArchived?: boolean },
  ) =>
    daemonRequest(DaemonMethods.listEnvironments, {
      projectId,
      ...options,
    }),
  createEnvironment: (params: Record<string, unknown>) =>
    daemonRequest(DaemonMethods.createEnvironment, params),
  updateEnvironment: (environmentId: string, patch: Record<string, unknown>) =>
    daemonRequest(DaemonMethods.updateEnvironment, {
      environmentId,
      ...patch,
    }),
  archiveEnvironment: (environmentId: string) =>
    daemonRequest(DaemonMethods.archiveEnvironment, { environmentId }),
  listFindings: (params: Record<string, unknown>) =>
    daemonRequest(DaemonMethods.listFindings, params),
  getFinding: (findingId: string) =>
    daemonRequest(DaemonMethods.getFinding, { findingId }),
  getFindingSummary: (params: Record<string, unknown>) =>
    daemonRequest(DaemonMethods.getFindingSummary, params),
  scanFindings: (params: Record<string, unknown>) =>
    daemonRequest<OperationRecord>(DaemonMethods.scanFindings, params),
  acknowledgeFinding: (findingId: string, comment?: string) =>
    daemonRequest(DaemonMethods.acknowledgeFinding, { findingId, comment }),
  dismissFinding: (findingId: string, reason?: string) =>
    daemonRequest(DaemonMethods.dismissFinding, { findingId, reason }),
  suppressFinding: (findingId: string, reason?: string) =>
    daemonRequest(DaemonMethods.suppressFinding, { findingId, reason }),
  listIntegrations: (projectId?: string) =>
    daemonRequest(DaemonMethods.listIntegrations, { projectId }),
  listPlugins: () => daemonRequest(DaemonMethods.listPlugins),
  listOperations: (params?: Record<string, unknown>) =>
    daemonRequest<OperationRecord[]>(DaemonMethods.listOperations, params ?? {}),
  listApprovals: (params?: Record<string, unknown>) =>
    daemonRequest<ApprovalRequestRecord[]>(
      DaemonMethods.listApprovals,
      params ?? {},
    ),
  decideApproval: (params: {
    approvalId: string;
    decision: "approved" | "denied";
    rememberScope?: boolean;
  }) => daemonRequest(DaemonMethods.decideApproval, params),
  listMcpClients: () =>
    daemonRequest<LocalClientPublicView[]>(DaemonMethods.listMcpClients),
  createMcpClient: (params: Record<string, unknown>) =>
    daemonRequest(DaemonMethods.createMcpClient, params),
  revokeMcpClient: (clientId: string) =>
    daemonRequest(DaemonMethods.revokeMcpClient, { clientId }),
  rotateMcpClientCredential: (clientId: string) =>
    daemonRequest<{ clientId: string; credential: string }>(
      DaemonMethods.rotateMcpClientCredential,
      { clientId },
    ),
  listAvailableCapabilities: () =>
    daemonRequest<{ permissions: string[]; methods: string[] }>(
      DaemonMethods.listAvailableCapabilities,
    ),
  listMcpAuditEvents: (limit?: number) =>
    daemonRequest(DaemonMethods.listMcpAuditEvents, { limit }),
  listConfigurationKeys: (projectId: string) =>
    daemonRequest(DaemonMethods.listConfigurationKeys, { projectId }),
  getEnvironmentConfiguration: (projectId: string, environmentId: string) =>
    daemonRequest(DaemonMethods.getEnvironmentConfiguration, {
      projectId,
      environmentId,
    }),
  setConfigurationValue: (params: Record<string, unknown>) =>
    daemonRequest(DaemonMethods.setConfigurationValue, params),
  setSensitiveConfigurationValue: (params: Record<string, unknown>) =>
    daemonRequest(DaemonMethods.setSensitiveConfigurationValue, params),
  syncProject: (projectId: string) =>
    daemonRequest(DaemonMethods.syncProject, { projectId }),
};

export type DesktopDaemon = typeof desktopDaemon;
