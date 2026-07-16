import type {
  InstalledPluginRecord,
  PluginConnectionRecord,
  PluginPermissionGrantRecord,
} from "@rayvan/local-database";

import { desktopDaemon, daemonRequest } from "../daemon/client.js";
import type {
  CreateIntegrationConnectionInput,
  GrantIntegrationPermissionsInput,
  PluginIntegrationsGateway,
} from "./types.js";

function toInstalledPlugin(plugin: {
  id?: string;
  pluginId?: string;
  name?: string;
  version?: string;
  status?: string;
  host?: string;
  enabled?: boolean;
  publisher?: string;
  description?: string;
  presentation?: InstalledPluginRecord["manifestSnapshot"]["presentation"];
}): InstalledPluginRecord | null {
  const pluginId = plugin.pluginId ?? plugin.id;
  if (!pluginId) {
    return null;
  }
  const name = plugin.name?.trim() || pluginId;
  const version = plugin.version ?? "0.0.0";
  const now = new Date().toISOString();
  const initials = name
    .split(/[\s-_]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("")
    .slice(0, 2) || pluginId.slice(0, 2).toUpperCase();

  return {
    id: pluginId,
    pluginId,
    pluginVersion: version,
    manifestVersion: "1",
    rayvanApiVersion: "1",
    source: { type: "built_in" },
    status: plugin.status === "disabled" ? "disabled" : "installed",
    enabled: plugin.enabled !== false && plugin.status !== "unavailable",
    installedAt: now,
    updatedAt: now,
    manifestSnapshot: {
      id: pluginId,
      name,
      description:
        plugin.description ??
        `Daemon plugin (${plugin.host ?? "unknown host"})`,
      version,
      publisher: plugin.publisher ?? "rayvan",
      rayvanApiVersion: "1",
      capabilities: [],
      permissions: [],
      resourceTypes: [],
      presentation: plugin.presentation ?? {
        icon: {
          iconId: pluginId,
          initials,
          label: name,
        },
        theme: {
          surface: "neutral",
          foregroundMode: "dark",
        },
        supportsMultipleConnections: false,
      },
    },
  };
}

/**
 * Daemon-backed integrations gateway. Lists plugins/connections from
 * `rayvand`; create/grant throw until richer plugin APIs are wired.
 */
export function createDaemonPluginIntegrationsGateway(): PluginIntegrationsGateway {
  return {
    async ensureProjectSeeded(): Promise<void> {},

    async listInstalledPlugins(): Promise<InstalledPluginRecord[]> {
      const plugins = (await desktopDaemon.listPlugins()) as Array<{
        id?: string;
        pluginId?: string;
        name?: string;
        version?: string;
        status?: string;
        host?: string;
        enabled?: boolean;
        publisher?: string;
        description?: string;
        presentation?: InstalledPluginRecord["manifestSnapshot"]["presentation"];
      }>;
      return plugins
        .map((plugin) => toInstalledPlugin(plugin))
        .filter((plugin): plugin is InstalledPluginRecord => plugin !== null);
    },

    async getInstalledPlugin(
      installedPluginId: string,
    ): Promise<InstalledPluginRecord | undefined> {
      const all = await this.listInstalledPlugins();
      return all.find(
        (plugin) =>
          plugin.id === installedPluginId ||
          plugin.pluginId === installedPluginId,
      );
    },

    async listConnectionsByProject(
      projectId: string,
    ): Promise<PluginConnectionRecord[]> {
      const integrations = (await desktopDaemon.listIntegrations(
        projectId,
      )) as Array<{
        id: string;
        pluginId: string;
        projectId: string;
        name: string;
        status: string;
        lastSuccessfulSyncAt?: string;
      }>;
      const now = new Date().toISOString();
      return integrations.map(
        (integration) =>
          ({
            id: integration.id,
            pluginId: integration.pluginId,
            installedPluginId: integration.pluginId,
            projectId: integration.projectId,
            name: integration.name,
            status: integration.status,
            createdAt: now,
            updatedAt: now,
            lastSuccessfulSyncAt: integration.lastSuccessfulSyncAt,
          }) as PluginConnectionRecord,
      );
    },

    async getConnection(
      connectionId: string,
    ): Promise<PluginConnectionRecord | undefined> {
      const connection = await daemonRequest("integrations.get", {
        integrationId: connectionId,
      });
      return (connection as PluginConnectionRecord | null) ?? undefined;
    },

    async listPermissionGrants(
      connectionId: string,
    ): Promise<PluginPermissionGrantRecord[]> {
      void connectionId;
      return [];
    },

    async createConnection(
      input: CreateIntegrationConnectionInput,
    ): Promise<PluginConnectionRecord> {
      void input;
      throw new Error(
        "Creating integrations through the daemon is not wired in the desktop gateway yet",
      );
    },

    async grantPermissions(
      input: GrantIntegrationPermissionsInput,
    ): Promise<PluginPermissionGrantRecord[]> {
      void input;
      throw new Error(
        "Granting plugin permissions through the daemon is not wired in the desktop gateway yet",
      );
    },

    async markConnected(connectionId: string): Promise<PluginConnectionRecord> {
      const connection = await this.getConnection(connectionId);
      if (!connection) {
        throw new Error(`Connection ${connectionId} not found`);
      }
      return connection;
    },

    async disconnectConnection(
      connectionId: string,
    ): Promise<PluginConnectionRecord> {
      const connection = await this.getConnection(connectionId);
      if (!connection) {
        throw new Error(`Connection ${connectionId} not found`);
      }
      return connection;
    },
  };
}
