import type {
  ConfigurationEntryId,
  EnvironmentId,
  IntegrationId,
} from "../ids/index.js";
import type { ConfigurationKey } from "./configuration-key.js";
import type { ConfigurationOccurrence } from "./configuration-occurrence.js";

/**
 * Legacy snapshot entry shape used by config-engine compare/drift helpers.
 * Prefer ConfigurationKey + ConfigurationOccurrence as the stored model;
 * project occurrences into this shape via `toConfigurationEntry`.
 *
 * Secret values are never stored in plain `value` fields.
 * Use valueFingerprint for drift detection without exposing secrets.
 */
export interface ConfigurationEntry {
  id: ConfigurationEntryId;
  environmentId: EnvironmentId;
  integrationId: IntegrationId;
  key: string;
  isSecret: boolean;
  isRequired?: boolean;
  description?: string;
  valueFingerprint?: string;
}

export interface ConfigurationSnapshot {
  environmentId: EnvironmentId;
  integrationId: IntegrationId;
  entries: ConfigurationEntry[];
  collectedAt: string;
}

/**
 * Compatibility projection: occurrence (+ key metadata) → ConfigurationEntry.
 * Requires an environmentId on the occurrence.
 *
 * Fingerprints are omitted for locked / name_only access so legacy
 * compare/drift helpers cannot treat inaccessible values as matching.
 * Prefer `buildConfigurationMatrix` for occurrence-aware comparison.
 */
export function toConfigurationEntry(
  key: ConfigurationKey,
  occurrence: ConfigurationOccurrence,
  integrationId: IntegrationId,
): ConfigurationEntry | undefined {
  if (!occurrence.environmentId) {
    return undefined;
  }
  const comparableAccess =
    occurrence.valueAccess === "readable" ||
    occurrence.valueAccess === "masked";
  return {
    id: occurrence.id as unknown as ConfigurationEntryId,
    environmentId: occurrence.environmentId,
    integrationId,
    key: key.name,
    isSecret: key.sensitive,
    isRequired: key.required,
    description: key.description,
    valueFingerprint: comparableAccess
      ? occurrence.valueFingerprint
      : undefined,
  };
}
