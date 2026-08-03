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
  trustStatus?: string;
  trustLabel?: string;
  presentation?: InstalledPluginRecord["manifestSnapshot"]["presentation"];
  setup?: InstalledPluginRecord["manifestSnapshot"]["setup"];
}): InstalledPluginRecord | null {
  const pluginId = plugin.pluginId ?? plugin.id;
  if (!pluginId) {
    return null;
  }
  const installedId = plugin.id ?? pluginId;
  const name = plugin.name?.trim() || pluginId;
  const version = plugin.version ?? "0.0.0";
  const now = new Date().toISOString();
  const initials =
    name
      .split(/[\s-_]+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase() ?? "")
      .join("")
      .slice(0, 2) || pluginId.slice(0, 2).toUpperCase();

  const isPackage = plugin.host === "out_of_process";
  const trustStatus =
    plugin.trustStatus === "signed" ||
    plugin.trustStatus === "unsigned_development"
      ? plugin.trustStatus
      : undefined;

  return {
    id: installedId,
    pluginId,
    pluginVersion: version,
    manifestVersion: "1",
    rayvanApiVersion: "1",
    source: isPackage
      ? {
          type: "package",
          packageId: pluginId,
          trustStatus,
          trustLabel: plugin.trustLabel,
          hostKind: "out_of_process",
        }
      : { type: "built_in" },
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
          iconId: pluginId.includes("github") ? "github" : pluginId,
          initials,
          label: name,
        },
        theme: {
          surface: "neutral",
          foregroundMode: "dark",
        },
        supportsMultipleConnections: false,
      },
      setup: plugin.setup,
    },
  };
}

function toConnectionRecord(integration: {
  id: string;
  pluginId: string;
  installedPluginId?: string;
  projectId: string;
  name: string;
  status: string;
  createdAt?: string;
  updatedAt?: string;
  lastSuccessfulSyncAt?: string;
}): PluginConnectionRecord {
  const now = new Date().toISOString();
  return {
    id: integration.id,
    pluginId: integration.pluginId,
    installedPluginId: integration.installedPluginId ?? integration.pluginId,
    projectId: integration.projectId,
    name: integration.name,
    status: integration.status as PluginConnectionRecord["status"],
    createdAt: integration.createdAt ?? now,
    updatedAt: integration.updatedAt ?? now,
    lastSuccessfulSyncAt: integration.lastSuccessfulSyncAt,
    metadata: {},
    schemaVersion: "1",
  };
}

/**
 * Daemon-backed integrations gateway. Lists plugins/connections from
 * `rayvand`. Create/setup go through plugin daemon methods; permission
 * grants are not yet persisted via daemon RPC (accepted as a soft no-op).
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
        trustStatus?: string;
        trustLabel?: string;
        presentation?: InstalledPluginRecord["manifestSnapshot"]["presentation"];
        setup?: InstalledPluginRecord["manifestSnapshot"]["setup"];
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
        installedPluginId?: string;
        projectId: string;
        name: string;
        status: string;
        createdAt?: string;
        updatedAt?: string;
        lastSuccessfulSyncAt?: string;
      }>;
      return integrations.map((integration) => toConnectionRecord(integration));
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
      const installed = await this.getInstalledPlugin(input.installedPluginId);
      if (!installed) {
        throw new Error(`Installed plugin ${input.installedPluginId} not found`);
      }

      const authMethods = installed.manifestSnapshot.setup?.authMethods ?? [];
      const wantsPat =
        Boolean(input.secretToken?.trim()) &&
        (input.authMethod === "pat" || authMethods.includes("pat"));

      if (wantsPat && input.secretToken) {
        const session = (await desktopDaemon.startPluginSetup({
          pluginId: installed.pluginId,
          projectId: input.projectId,
          authMethod: "pat",
        })) as { id: string };

        await desktopDaemon.stepPluginSetup({
          sessionId: session.id,
          stepId: "pat-input",
          authMethod: "pat",
          connectionName: input.name,
          secretToken: input.secretToken.trim(),
          statePatch: { connectionName: input.name },
        });

        await desktopDaemon.completePluginSetup(session.id);

        const connections = await this.listConnectionsByProject(input.projectId);
        const created = connections
          .filter((connection) => connection.pluginId === installed.pluginId)
          .sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
        if (!created) {
          throw new Error(
            "GitHub connection was created but could not be loaded",
          );
        }
        return created;
      }

      return (await desktopDaemon.createPluginConnection({
        pluginId: installed.pluginId,
        projectId: input.projectId,
        name: input.name,
        metadata: input.metadata,
      })) as PluginConnectionRecord;
    },

    async grantPermissions(
      input: GrantIntegrationPermissionsInput,
    ): Promise<PluginPermissionGrantRecord[]> {
      // Daemon permission-grant RPC is not wired yet; UI still collects the
      // intended grants so the configure step stays consistent with fixtures.
      void input;
      return [];
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
