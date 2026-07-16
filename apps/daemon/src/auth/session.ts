import {
  BUILT_IN_PERMISSION_PROFILES,
  type BuiltInPermissionProfileId,
  type DaemonClientType,
  type LocalClientRecord,
  type McpPermission,
  type RayvanActor,
} from "@rayvan/daemon-contracts";

import { DaemonAppError } from "../errors.js";

export interface SessionContext {
  sessionId: string;
  clientType: DaemonClientType;
  clientVersion: string;
  client?: LocalClientRecord;
  actor: RayvanActor;
  permissions: ReadonlySet<McpPermission>;
  projectScopes: string[] | "*";
  environmentScopes?: string[] | "*";
  subscribed: boolean;
}

export function permissionsForProfile(profileId: string): ReadonlySet<McpPermission> {
  if (profileId in BUILT_IN_PERMISSION_PROFILES) {
    return new Set(
      BUILT_IN_PERMISSION_PROFILES[
        profileId as Exclude<BuiltInPermissionProfileId, "custom">
      ],
    );
  }
  return new Set();
}

export function requirePermission(
  session: SessionContext,
  permission: McpPermission,
): void {
  if (!session.permissions.has(permission)) {
    throw new DaemonAppError("PERMISSION_DENIED", `Missing permission: ${permission}`, {
      details: { permission },
    });
  }
}

export function requireProjectScope(session: SessionContext, projectId: string): void {
  if (session.client?.permissionProfileId === "administrator") {
    return;
  }
  if (session.projectScopes === "*") {
    return;
  }
  if (!session.projectScopes.includes(projectId)) {
    throw new DaemonAppError(
      "PROJECT_SCOPE_DENIED",
      `Client is not scoped to project ${projectId}`,
      { details: { projectId } },
    );
  }
}

export function requireEnvironmentScope(
  session: SessionContext,
  environmentId: string,
): void {
  if (session.client?.permissionProfileId === "administrator") {
    return;
  }
  if (!session.environmentScopes || session.environmentScopes === "*") {
    return;
  }
  if (!session.environmentScopes.includes(environmentId)) {
    throw new DaemonAppError(
      "ENVIRONMENT_SCOPE_DENIED",
      `Client is not scoped to environment ${environmentId}`,
      { details: { environmentId } },
    );
  }
}
