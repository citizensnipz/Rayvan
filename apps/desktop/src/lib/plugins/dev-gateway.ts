import {
  DevelopmentMemoryCredentialStore,
  PluginConnectionService,
  PluginInstallationService,
  PluginPermissionService,
  createInMemoryPluginPersistence,
} from "@rayvan/local-database";
import type {
  InstalledPluginRecord,
  PluginConnectionRecord,
  PluginPermissionGrantRecord,
} from "@rayvan/local-database";

import { ensureCatalogInstalled, seedProjectConnections } from "./dev-fixtures.js";
import type {
  CreateIntegrationConnectionInput,
  GrantIntegrationPermissionsInput,
  PluginIntegrationsGateway,
} from "./types.js";

/**
 * DEVELOPMENT ONLY gateway implementation.
 *
 * Backs the Integrations UI with in-memory plugin persistence
 * (`createInMemoryPluginPersistence`) seeded with fixture data — never
 * sqlite/better-sqlite3, and never real plugin execution or provider APIs.
 * Each call to `createDevPluginIntegrationsGateway()` returns a fresh,
 * isolated instance (no module-level singleton) so tests never share state.
 */
export function createDevPluginIntegrationsGateway(): PluginIntegrationsGateway {
  const persistence = createInMemoryPluginPersistence();
  const credentialStore = new DevelopmentMemoryCredentialStore();

  const installationService = new PluginInstallationService(
    persistence.installedPlugins,
  );
  const connectionService = new PluginConnectionService(
    persistence.installedPlugins,
    persistence.connections,
    persistence.credentialReferences,
    persistence.permissionGrants,
    persistence.resourceBindings,
    credentialStore,
  );
  const permissionService = new PluginPermissionService(
    persistence.connections,
    persistence.permissionGrants,
  );

  let catalogPromise: Promise<Map<string, InstalledPluginRecord>> | null = null;
  const seededProjectIds = new Set<string>();

  function ensureCatalog(): Promise<Map<string, InstalledPluginRecord>> {
    catalogPromise ??= ensureCatalogInstalled(persistence.installedPlugins);
    return catalogPromise;
  }

  return {
    async ensureProjectSeeded(projectId: string): Promise<void> {
      const installedByPluginId = await ensureCatalog();
      if (seededProjectIds.has(projectId)) {
        return;
      }
      seededProjectIds.add(projectId);
      await seedProjectConnections(
        { connections: persistence.connections, permissionGrants: persistence.permissionGrants },
        projectId,
        installedByPluginId,
      );
    },

    async listInstalledPlugins(): Promise<InstalledPluginRecord[]> {
      await ensureCatalog();
      return installationService.list();
    },

    async getInstalledPlugin(
      installedPluginId: string,
    ): Promise<InstalledPluginRecord | undefined> {
      return persistence.installedPlugins.getById(installedPluginId);
    },

    async listConnectionsByProject(
      projectId: string,
    ): Promise<PluginConnectionRecord[]> {
      return connectionService.listByProjectId(projectId);
    },

    async getConnection(
      connectionId: string,
    ): Promise<PluginConnectionRecord | undefined> {
      return connectionService.getById(connectionId);
    },

    async listPermissionGrants(
      connectionId: string,
    ): Promise<PluginPermissionGrantRecord[]> {
      return permissionService.listActive(connectionId);
    },

    async createConnection(
      input: CreateIntegrationConnectionInput,
    ): Promise<PluginConnectionRecord> {
      const connection = await connectionService.create({
        installedPluginId: input.installedPluginId,
        projectId: input.projectId,
        name: input.name,
        externalAccountId: input.externalAccountId,
        externalAccountName: input.externalAccountName,
        metadata: input.metadata ?? {},
      });
      return connectionService.markConnected(connection.id);
    },

    async grantPermissions(
      input: GrantIntegrationPermissionsInput,
    ): Promise<PluginPermissionGrantRecord[]> {
      return permissionService.grant({
        pluginId: input.pluginId,
        connectionId: input.connectionId,
        permissions: input.permissions,
        projectId: input.projectId,
        grantedBy: input.grantedBy,
        reason: input.reason,
      });
    },

    async markConnected(connectionId: string): Promise<PluginConnectionRecord> {
      return connectionService.markConnected(connectionId);
    },

    async disconnectConnection(
      connectionId: string,
    ): Promise<PluginConnectionRecord> {
      return connectionService.disconnect(connectionId);
    },
  };
}
