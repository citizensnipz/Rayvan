/** Daemon IPC protocol version. Handshake rejects incompatible clients. */
export const DAEMON_PROTOCOL_VERSION = "1" as const;

export type DaemonClientType =
  "desktop" | "mcp" | "cli" | "test" | "future_cloud_bridge";

export interface DaemonHandshakeRequest {
  protocolVersion: string;
  clientType: DaemonClientType;
  clientVersion: string;
  clientId?: string;
  /** Opaque client credential; never logged or audited in plaintext. */
  clientCredential?: string;
}

export interface DaemonHandshakeResponse {
  protocolVersion: typeof DAEMON_PROTOCOL_VERSION;
  daemonVersion: string;
  sessionId: string;
  capabilities: string[];
  authenticatedClientId?: string;
  permissionProfileId?: string;
}

export interface DaemonRequestEnvelope {
  jsonrpc: "2.0";
  id: string;
  method: string;
  params?: unknown;
}

export interface DaemonSuccessResponse {
  jsonrpc: "2.0";
  id: string;
  result: unknown;
}

export interface DaemonErrorObject {
  code: number;
  message: string;
  data?: DaemonSerializedError;
}

export interface DaemonErrorResponse {
  jsonrpc: "2.0";
  id: string | null;
  error: DaemonErrorObject;
}

export type DaemonResponseEnvelope = DaemonSuccessResponse | DaemonErrorResponse;

/** Application-level request wrapper carried in JSON-RPC params when needed. */
export interface DaemonRequestMeta {
  requestId: string;
  protocolVersion: string;
  idempotencyKey?: string;
  correlationId?: string;
}

export type DaemonResponseStatus = "succeeded" | "failed";

export interface DaemonSucceededResponse<T> {
  requestId: string;
  status: "succeeded";
  data: T;
}

export interface DaemonFailedResponse {
  requestId: string;
  status: "failed";
  error: DaemonSerializedError;
}

export type DaemonResponse<T> = DaemonSucceededResponse<T> | DaemonFailedResponse;

export interface DaemonSerializedError {
  code: DaemonErrorCode;
  message: string;
  retryable: boolean;
  correlationId?: string;
  details?: Record<string, string | number | boolean | null>;
}

export type DaemonErrorCode =
  | "DAEMON_UNAVAILABLE"
  | "DAEMON_VERSION_MISMATCH"
  | "CLIENT_NOT_REGISTERED"
  | "CLIENT_REVOKED"
  | "PERMISSION_DENIED"
  | "PROJECT_SCOPE_DENIED"
  | "ENVIRONMENT_SCOPE_DENIED"
  | "APPROVAL_REQUIRED"
  | "APPROVAL_DENIED"
  | "REVISION_CONFLICT"
  | "OPERATION_ALREADY_RUNNING"
  | "PLUGIN_UNAVAILABLE"
  | "PLUGIN_PERMISSION_DENIED"
  | "PLAN_STALE"
  | "PLAN_NOT_APPROVED"
  | "SECRET_ACCESS_DENIED"
  | "RESOURCE_LOCKED"
  | "PROVIDER_OPERATION_FAILED"
  | "VERIFICATION_FAILED"
  | "DATABASE_ERROR"
  | "VALIDATION_FAILED"
  | "NOT_FOUND"
  | "CANCELLED"
  | "INTERNAL_ERROR"
  | "METHOD_NOT_FOUND"
  | "UNAUTHENTICATED";

/** JSON-RPC application error codes (custom range). */
export const DAEMON_RPC_ERROR_CODES = {
  APPLICATION: -32000,
  UNAUTHENTICATED: -32001,
  PERMISSION_DENIED: -32002,
  VERSION_MISMATCH: -32003,
} as const;
