import {
  RAYVAN_PLUGIN_API_VERSION,
  type PluginManifest,
} from "@rayvan/plugin-sdk";

import { PluginDomainError, PluginNotFoundError } from "../errors.js";
import type { InstalledPluginRecord } from "../models.js";
import type { InstalledPluginRepository } from "../repositories/types.js";

export class PluginInstallationService {
  constructor(private readonly installedPlugins: InstalledPluginRepository) {}

  async getByPluginId(
    pluginId: string,
  ): Promise<InstalledPluginRecord | undefined> {
    return this.installedPlugins.getByPluginId(pluginId);
  }

  async list(): Promise<InstalledPluginRecord[]> {
    return this.installedPlugins.list();
  }

  async setEnabled(id: string, enabled: boolean): Promise<void> {
    const existing = await this.installedPlugins.getById(id);
    if (!existing) {
      throw new PluginNotFoundError(id);
    }
    if (
      enabled &&
      (existing.status === "incompatible" || existing.status === "missing")
    ) {
      throw new PluginDomainError(
        `Cannot enable plugin in status ${existing.status}`,
      );
    }
    await this.installedPlugins.setEnabled(id, enabled);
  }

  async disable(id: string): Promise<void> {
    await this.setEnabled(id, false);
  }

  /**
   * Soft-uninstall: disable and mark status `disabled` while preserving
   * installation history, connections, and audit records.
   */
  async uninstall(id: string): Promise<InstalledPluginRecord> {
    const existing = await this.installedPlugins.getById(id);
    if (!existing) {
      throw new PluginNotFoundError(id);
    }
    const next: InstalledPluginRecord = {
      ...existing,
      enabled: false,
      status: "disabled",
      updatedAt: new Date().toISOString(),
    };
    await this.installedPlugins.save(next);
    return next;
  }

  /**
   * Reconcile built-in plugin manifests with installed records at startup.
   * Never deletes history; never auto-enables incompatible or recovered plugins.
   */
  async reconcileBuiltIns(
    plugins: Array<{ manifest: PluginManifest }>,
  ): Promise<InstalledPluginRecord[]> {
    const now = new Date().toISOString();
    const availableIds = new Set(plugins.map((plugin) => plugin.manifest.id));
    const results: InstalledPluginRecord[] = [];

    for (const { manifest } of plugins) {
      const existing = await this.installedPlugins.getByPluginId(manifest.id);
      const apiSupported = manifest.rayvanApiVersion === RAYVAN_PLUGIN_API_VERSION;

      if (!existing) {
        const record: InstalledPluginRecord = {
          id: crypto.randomUUID(),
          pluginId: manifest.id,
          pluginVersion: manifest.version,
          manifestVersion: manifest.version,
          rayvanApiVersion: manifest.rayvanApiVersion,
          source: { type: "built_in" },
          status: apiSupported ? "installed" : "incompatible",
          enabled: apiSupported,
          installedAt: now,
          updatedAt: now,
          lastLoadedAt: now,
          manifestSnapshot: structuredClone(manifest),
        };
        await this.installedPlugins.save(record);
        results.push(record);
        continue;
      }

      const next: InstalledPluginRecord = {
        ...existing,
        pluginVersion: manifest.version,
        manifestVersion: manifest.version,
        rayvanApiVersion: manifest.rayvanApiVersion,
        manifestSnapshot: structuredClone(manifest),
        lastLoadedAt: now,
        updatedAt: now,
        status: "installed",
        enabled: existing.enabled,
      };

      if (!apiSupported) {
        next.enabled = false;
        next.status = "incompatible";
      } else if (
        existing.status === "missing" ||
        existing.status === "incompatible"
      ) {
        // Recover availability without auto-enabling.
        next.status = "installed";
        next.enabled = false;
      } else if (!existing.enabled || existing.status === "disabled") {
        next.status = "disabled";
        next.enabled = false;
      }

      await this.installedPlugins.save(next);
      results.push(next);
    }

    const all = await this.installedPlugins.list();
    for (const record of all) {
      if (
        record.source.type === "built_in" &&
        !availableIds.has(record.pluginId) &&
        record.status !== "missing"
      ) {
        const missing: InstalledPluginRecord = {
          ...record,
          status: "missing",
          enabled: false,
          updatedAt: now,
        };
        await this.installedPlugins.save(missing);
        results.push(missing);
      }
    }

    return results;
  }
}
