import type { BuiltInPermissionProfileId, McpApprovalPolicy } from "./permissions.js";

export type LocalClientType = "desktop" | "mcp" | "cli";

export type LocalClientStatus = "active" | "revoked" | "expired";

export const BUILT_IN_LOCAL_CLIENT_IDS = {
  desktop: "rayvan-desktop",
  cli: "rayvan-cli",
} as const;

export interface LocalClientRecord {
  id: string;
  name: string;
  type: LocalClientType;
  status: LocalClientStatus;
  permissionProfileId: BuiltInPermissionProfileId | string;
  projectScopes: string[];
  environmentScopes?: string[];
  approvalPolicy: McpApprovalPolicy;
  createdAt: string;
  lastConnectedAt?: string;
  lastActivityAt?: string;
  /** Reference into secure credential store — never the raw token. */
  credentialReferenceId: string;
}

export interface LocalClientPublicView {
  id: string;
  name: string;
  type: LocalClientType;
  status: LocalClientStatus;
  permissionProfileId: string;
  projectScopes: string[];
  environmentScopes?: string[];
  approvalPolicy: McpApprovalPolicy;
  createdAt: string;
  lastConnectedAt?: string;
  lastActivityAt?: string;
  connected: boolean;
}
