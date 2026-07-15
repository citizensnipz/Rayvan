import type {
  PluginExecutionEvent,
  PluginExecutionEventSink,
  PluginPermission,
  PluginPermissionResolveContext,
  PluginPermissionResolver,
} from "@rayvan/plugin-sdk";

import type {
  InstalledPluginRepository,
  PluginConnectionRepository,
  PluginExecutionHistoryRepository,
  PluginPermissionGrantRepository,
} from "../repositories/types.js";
import type { PluginExecutionHistoryRecord } from "../models.js";

/**
 * Resolves active grants from persistence.
 * Disconnected / revoked connections yield no grants.
 */
export class PersistentPluginPermissionResolver
  implements PluginPermissionResolver
{
  constructor(
    private readonly connections: PluginConnectionRepository,
    private readonly grants: PluginPermissionGrantRepository,
    private readonly options?: { connectionId?: string },
  ) {}

  async resolve(
    context: PluginPermissionResolveContext,
  ): Promise<readonly PluginPermission[]> {
    const connectionId = this.options?.connectionId;
    if (!connectionId) {
      return [];
    }

    const connection = await this.connections.getById(connectionId);
    if (!connection || connection.pluginId !== context.pluginId) {
      return [];
    }
    if (
      connection.status === "disconnected" ||
      connection.status === "revoked"
    ) {
      return [];
    }

    const active = await this.grants.listActiveByConnectionId(connectionId);
    const matched = active.filter((grant) => {
      if (grant.projectId && context.projectId && grant.projectId !== context.projectId) {
        return false;
      }
      if (
        grant.environmentId &&
        context.environmentId &&
        grant.environmentId !== context.environmentId
      ) {
        return false;
      }
      // Environment-scoped grants must not broaden to project-only requests.
      if (grant.environmentId && !context.environmentId) {
        return false;
      }
      if (grant.projectId && !context.projectId) {
        return false;
      }
      return true;
    });

    return [...new Set(matched.map((grant) => grant.permission))];
  }
}

export class PersistentPluginExecutionEventSink
  implements PluginExecutionEventSink
{
  constructor(
    private readonly history: PluginExecutionHistoryRepository,
    private readonly options?: { connectionId?: string },
  ) {}

  async record(event: PluginExecutionEvent): Promise<void> {
    const record: PluginExecutionHistoryRecord = {
      id: crypto.randomUUID(),
      executionId: event.executionId,
      pluginId: event.pluginId,
      pluginVersion: event.pluginVersion,
      capability: event.capability,
      status: event.status,
      startedAt: event.startedAt,
      finishedAt: event.finishedAt,
      durationMs: event.durationMs,
      actor: event.actor,
      projectId: event.projectId,
      environmentId: event.environmentId,
      resourceId: event.resourceId,
      connectionId: this.options?.connectionId,
      reason: event.reason,
      errorCode: event.errorCode,
      errorMessage: event.errorMessage,
      warningCount: event.warningCount,
      recordedAt: new Date().toISOString(),
    };
    await this.history.append(record);
  }
}

export interface PluginExecutionGuardResult {
  ok: boolean;
  reason?: string;
}

/**
 * Host-side pre-execution checks. Does not live inside PluginExecutionService.
 */
export class PluginExecutionGuard {
  constructor(
    private readonly installedPlugins: InstalledPluginRepository,
    private readonly connections: PluginConnectionRepository,
  ) {}

  async assertExecutable(input: {
    pluginId: string;
    connectionId?: string;
  }): Promise<PluginExecutionGuardResult> {
    const installed = await this.installedPlugins.getByPluginId(input.pluginId);
    if (!installed) {
      return { ok: false, reason: "plugin_not_installed" };
    }
    if (!installed.enabled || installed.status !== "installed") {
      return { ok: false, reason: `plugin_not_executable:${installed.status}` };
    }

    if (input.connectionId) {
      const connection = await this.connections.getById(input.connectionId);
      if (!connection) {
        return { ok: false, reason: "connection_not_found" };
      }
      if (connection.pluginId !== input.pluginId) {
        return { ok: false, reason: "connection_plugin_mismatch" };
      }
      if (
        connection.status === "disconnected" ||
        connection.status === "revoked"
      ) {
        return { ok: false, reason: `connection_inactive:${connection.status}` };
      }
    }

    return { ok: true };
  }
}

export function createPersistentPermissionResolver(input: {
  connections: PluginConnectionRepository;
  grants: PluginPermissionGrantRepository;
  connectionId?: string;
}): PersistentPluginPermissionResolver {
  return new PersistentPluginPermissionResolver(
    input.connections,
    input.grants,
    { connectionId: input.connectionId },
  );
}

export function createPersistentExecutionEventSink(input: {
  history: PluginExecutionHistoryRepository;
  connectionId?: string;
}): PersistentPluginExecutionEventSink {
  return new PersistentPluginExecutionEventSink(input.history, {
    connectionId: input.connectionId,
  });
}
