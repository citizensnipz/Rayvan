import type { PluginSetupSessionRecord } from "../models-setup.js";

export interface PluginSetupSessionRepository {
  save(record: PluginSetupSessionRecord): Promise<void>;
  getById(id: string): Promise<PluginSetupSessionRecord | undefined>;
  listByPluginId(pluginId: string): Promise<PluginSetupSessionRecord[]>;
  listActiveByProjectId(projectId: string): Promise<PluginSetupSessionRecord[]>;
}

export class InMemoryPluginSetupSessionRepository
  implements PluginSetupSessionRepository
{
  private readonly byId = new Map<string, PluginSetupSessionRecord>();

  async save(record: PluginSetupSessionRecord): Promise<void> {
    this.byId.set(record.id, structuredClone(record));
  }

  async getById(id: string): Promise<PluginSetupSessionRecord | undefined> {
    const record = this.byId.get(id);
    return record ? structuredClone(record) : undefined;
  }

  async listByPluginId(pluginId: string): Promise<PluginSetupSessionRecord[]> {
    return [...this.byId.values()]
      .filter((record) => record.pluginId === pluginId)
      .map((record) => structuredClone(record));
  }

  async listActiveByProjectId(
    projectId: string,
  ): Promise<PluginSetupSessionRecord[]> {
    return [...this.byId.values()]
      .filter(
        (record) =>
          record.projectId === projectId && record.status === "active",
      )
      .map((record) => structuredClone(record));
  }
}
