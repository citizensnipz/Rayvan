import type { PluginExecutionActor, PluginPermission } from "@rayvan/plugin-sdk";

import {
  PluginConnectionNotFoundError,
  PluginDomainError,
} from "../errors.js";
import type { PluginPermissionGrantRecord } from "../models.js";
import type {
  PluginConnectionRepository,
  PluginPermissionGrantRepository,
} from "../repositories/types.js";

export interface GrantPermissionsInput {
  pluginId: string;
  connectionId: string;
  permissions: readonly PluginPermission[];
  projectId?: string;
  environmentId?: string;
  grantedBy: PluginExecutionActor;
  reason?: string;
}

export class PluginPermissionService {
  constructor(
    private readonly connections: PluginConnectionRepository,
    private readonly grants: PluginPermissionGrantRepository,
  ) {}

  async grant(input: GrantPermissionsInput): Promise<PluginPermissionGrantRecord[]> {
    const connection = await this.connections.getById(input.connectionId);
    if (!connection) {
      throw new PluginConnectionNotFoundError(input.connectionId);
    }
    if (
      connection.status === "disconnected" ||
      connection.status === "revoked"
    ) {
      throw new PluginDomainError(
        "Cannot grant permissions on a disconnected or revoked connection",
      );
    }
    if (connection.pluginId !== input.pluginId) {
      throw new PluginDomainError("pluginId does not match connection");
    }

    const now = new Date().toISOString();
    const records: PluginPermissionGrantRecord[] = input.permissions.map(
      (permission) => ({
        id: crypto.randomUUID(),
        pluginId: input.pluginId,
        connectionId: input.connectionId,
        permission,
        projectId: input.projectId,
        environmentId: input.environmentId,
        granted: true,
        grantedBy: input.grantedBy,
        grantedAt: now,
        reason: input.reason,
      }),
    );

    await this.grants.replaceActiveGrants({
      connectionId: input.connectionId,
      projectId: input.projectId,
      environmentId: input.environmentId,
      grants: records,
    });
    return records;
  }

  async revoke(input: {
    grantId: string;
    revokedBy: PluginExecutionActor;
    reason?: string;
  }): Promise<PluginPermissionGrantRecord> {
    const existing = await this.grants.getById(input.grantId);
    if (!existing) {
      throw new PluginDomainError(`Permission grant not found: ${input.grantId}`);
    }
    const updated: PluginPermissionGrantRecord = {
      ...existing,
      granted: false,
      revokedBy: input.revokedBy,
      revokedAt: new Date().toISOString(),
      reason: input.reason ?? existing.reason,
    };
    await this.grants.save(updated);
    return updated;
  }

  async listActive(
    connectionId: string,
  ): Promise<PluginPermissionGrantRecord[]> {
    return this.grants.listActiveByConnectionId(connectionId);
  }
}
