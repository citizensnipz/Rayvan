import {
  PluginConnectionNotFoundError,
  PluginDomainError,
  PluginNotFoundError,
} from "../errors.js";
import type { CredentialStore } from "../credentials/types.js";
import type {
  CredentialReferenceRecord,
  PluginConnectionRecord,
  PluginConnectionStatus,
} from "../models.js";
import type {
  CredentialReferenceRepository,
  InstalledPluginRepository,
  PluginConnectionRepository,
  PluginPermissionGrantRepository,
  ResourceBindingRepository,
} from "../repositories/types.js";

export interface CreateConnectionInput {
  installedPluginId: string;
  name: string;
  projectId?: string;
  externalAccountId?: string;
  externalAccountName?: string;
  providerBaseUrl?: string;
  metadata?: Record<string, unknown>;
  schemaVersion?: string;
  status?: PluginConnectionStatus;
}

export class PluginConnectionService {
  constructor(
    private readonly installedPlugins: InstalledPluginRepository,
    private readonly connections: PluginConnectionRepository,
    private readonly credentialReferences: CredentialReferenceRepository,
    private readonly permissionGrants: PluginPermissionGrantRepository,
    private readonly resourceBindings: ResourceBindingRepository,
    private readonly credentialStore: CredentialStore,
  ) {}

  async create(input: CreateConnectionInput): Promise<PluginConnectionRecord> {
    const installed = await this.installedPlugins.getById(
      input.installedPluginId,
    );
    if (!installed) {
      throw new PluginNotFoundError(input.installedPluginId);
    }
    if (!installed.enabled || installed.status !== "installed") {
      throw new PluginDomainError(
        "Cannot create connection for disabled or unavailable plugin",
      );
    }

    const now = new Date().toISOString();
    const record: PluginConnectionRecord = {
      id: crypto.randomUUID(),
      installedPluginId: installed.id,
      pluginId: installed.pluginId,
      projectId: input.projectId,
      name: input.name.trim(),
      status: input.status ?? "pending",
      externalAccountId: input.externalAccountId,
      externalAccountName: input.externalAccountName,
      providerBaseUrl: input.providerBaseUrl,
      metadata: input.metadata ?? {},
      schemaVersion: input.schemaVersion ?? "1",
      createdAt: now,
      updatedAt: now,
    };
    await this.connections.save(record);
    return record;
  }

  async attachCredentialReference(
    connectionId: string,
    reference: CredentialReferenceRecord,
  ): Promise<PluginConnectionRecord> {
    const connection = await this.requireConnection(connectionId);
    await this.credentialReferences.save(reference);
    const updated: PluginConnectionRecord = {
      ...connection,
      credentialReferenceId: reference.id,
      updatedAt: new Date().toISOString(),
    };
    await this.connections.save(updated);
    return updated;
  }

  async markConnected(connectionId: string): Promise<PluginConnectionRecord> {
    const connection = await this.requireConnection(connectionId);
    const now = new Date().toISOString();
    const updated: PluginConnectionRecord = {
      ...connection,
      status: "connected",
      lastAuthenticatedAt: now,
      updatedAt: now,
      lastErrorCode: undefined,
    };
    await this.connections.save(updated);
    return updated;
  }

  /**
   * Disconnect: soft-status only. Preserves resources, plans, and history.
   * Deletes credential material via CredentialStore and invalidates bindings.
   */
  async disconnect(connectionId: string): Promise<PluginConnectionRecord> {
    const connection = await this.requireConnection(connectionId);
    const now = new Date().toISOString();

    if (connection.credentialReferenceId) {
      const reference = await this.credentialReferences.getById(
        connection.credentialReferenceId,
      );
      if (reference) {
        await this.credentialStore.delete(reference);
      }
    }

    const refs = await this.credentialReferences.listByConnectionId(connectionId);
    for (const reference of refs) {
      await this.credentialStore.delete(reference);
    }

    const grants =
      await this.permissionGrants.listActiveByConnectionId(connectionId);
    for (const grant of grants) {
      await this.permissionGrants.save({
        ...grant,
        granted: false,
        revokedAt: now,
        reason: grant.reason ?? "connection disconnected",
      });
    }

    await this.resourceBindings.invalidateByConnectionId(connectionId);

    const updated: PluginConnectionRecord = {
      ...connection,
      status: "disconnected",
      credentialReferenceId: undefined,
      updatedAt: now,
    };
    await this.connections.save(updated);
    return updated;
  }

  async listByProjectId(projectId: string): Promise<PluginConnectionRecord[]> {
    return this.connections.listByProjectId(projectId);
  }

  async listByPluginId(pluginId: string): Promise<PluginConnectionRecord[]> {
    return this.connections.listByPluginId(pluginId);
  }

  async getById(id: string): Promise<PluginConnectionRecord | undefined> {
    return this.connections.getById(id);
  }

  private async requireConnection(
    id: string,
  ): Promise<PluginConnectionRecord> {
    const connection = await this.connections.getById(id);
    if (!connection) {
      throw new PluginConnectionNotFoundError(id);
    }
    return connection;
  }
}
