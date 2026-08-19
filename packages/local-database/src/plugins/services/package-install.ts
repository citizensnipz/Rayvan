import { randomUUID } from "node:crypto";

import type { PluginManifest } from "@rayvan/plugin-sdk";

import { PluginDomainError } from "../errors.js";
import type {
  InstalledPluginRecord,
  PluginHostKind,
  PluginPackageTrustStatus,
} from "../models.js";
import type { InstalledPluginRepository } from "../repositories/types.js";

export interface PackageInstallInput {
  manifest: PluginManifest;
  packagePath: string;
  installPath: string;
  binaryPath: string;
  trustStatus: PluginPackageTrustStatus;
  trustLabel: string;
  targetTriple?: string;
  signerFingerprint?: string;
  hostKind?: PluginHostKind;
  /** Soft-update: replace version while preserving installation id/history. */
  replaceExisting?: boolean;
}

export class PluginPackageInstallService {
  constructor(private readonly installedPlugins: InstalledPluginRepository) {}

  async installFromPackage(
    input: PackageInstallInput,
  ): Promise<InstalledPluginRecord> {
    const existing = await this.installedPlugins.getByPluginId(
      input.manifest.id,
    );
    const now = new Date().toISOString();

    if (existing && !input.replaceExisting) {
      throw new PluginDomainError(
        `Plugin ${input.manifest.id} is already installed; pass replaceExisting to update`,
      );
    }

    const record: InstalledPluginRecord = {
      id: existing?.id ?? randomUUID(),
      pluginId: input.manifest.id,
      pluginVersion: input.manifest.version,
      manifestVersion: input.manifest.version,
      rayvanApiVersion: input.manifest.rayvanApiVersion,
      source: {
        type: "package",
        packageId: input.manifest.id,
        packagePath: input.packagePath,
        installPath: input.installPath,
        binaryPath: input.binaryPath,
        trustStatus: input.trustStatus,
        trustLabel: input.trustLabel,
        targetTriple: input.targetTriple,
        hostKind: input.hostKind ?? "out_of_process",
        signerFingerprint: input.signerFingerprint,
      },
      status: "installed",
      enabled: true,
      installedAt: existing?.installedAt ?? now,
      updatedAt: now,
      lastLoadedAt: now,
      manifestSnapshot: structuredClone(input.manifest),
    };
    await this.installedPlugins.save(record);
    return record;
  }
}
