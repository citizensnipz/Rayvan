import type { PluginExecutionActor, PluginPermission } from "@rayvan/plugin-sdk";
import type {
  InstalledPluginRecord,
  PluginConnectionRecord,
  PluginPermissionGrantRecord,
} from "@rayvan/local-database";

/**
 * Host-side abstraction over plugin installation/connection/permission
 * persistence for the Integrations UI. Bound to `PluginConnectionRecord` +
 * `InstalledPluginRecord.manifestSnapshot.presentation` (never Core
 * `Integration`). Implementations must not execute plugin code.
 */
export interface CreateIntegrationConnectionInput {
  installedPluginId: string;
  projectId: string;
  name: string;
  externalAccountId?: string;
  externalAccountName?: string;
  metadata?: Record<string, unknown>;
}

export interface GrantIntegrationPermissionsInput {
  pluginId: string;
  connectionId: string;
  projectId: string;
  permissions: readonly PluginPermission[];
  grantedBy: PluginExecutionActor;
  reason?: string;
}

export interface PluginIntegrationsGateway {
  /**
   * Idempotently seeds development fixtures for a project the first time
   * it is seen. No-op for gateways backed by real persistence. Safe to call
   * on every mount.
   */
  ensureProjectSeeded(projectId: string): Promise<void>;

  listInstalledPlugins(): Promise<InstalledPluginRecord[]>;
  getInstalledPlugin(
    installedPluginId: string,
  ): Promise<InstalledPluginRecord | undefined>;

  listConnectionsByProject(
    projectId: string,
  ): Promise<PluginConnectionRecord[]>;
  getConnection(
    connectionId: string,
  ): Promise<PluginConnectionRecord | undefined>;

  listPermissionGrants(
    connectionId: string,
  ): Promise<PluginPermissionGrantRecord[]>;

  createConnection(
    input: CreateIntegrationConnectionInput,
  ): Promise<PluginConnectionRecord>;
  grantPermissions(
    input: GrantIntegrationPermissionsInput,
  ): Promise<PluginPermissionGrantRecord[]>;
  markConnected(connectionId: string): Promise<PluginConnectionRecord>;
  disconnectConnection(connectionId: string): Promise<PluginConnectionRecord>;
}
