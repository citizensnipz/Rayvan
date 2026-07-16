import type { RayvanActor } from "./actor.js";

export type DaemonEventType =
  | "project_changed"
  | "environment_changed"
  | "integration_changed"
  | "configuration_changed"
  | "finding_changed"
  | "operation_started"
  | "operation_progress"
  | "operation_completed"
  | "approval_requested"
  | "approval_decided"
  | "plugin_state_changed"
  | "daemon_status_changed"
  | "mcp_client_changed";

export interface DaemonEventBase {
  eventId: string;
  type: DaemonEventType;
  timestamp: string;
  schemaVersion: "1";
  projectId?: string;
  actor?: RayvanActor;
  correlationId?: string;
  payload: Record<string, unknown>;
}

export type DaemonEvent = DaemonEventBase;

/** Server → client notification (JSON-RPC style). */
export interface DaemonEventNotification {
  jsonrpc: "2.0";
  method: "daemon.event";
  params: DaemonEvent;
}
