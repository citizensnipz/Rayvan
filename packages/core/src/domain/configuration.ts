import type {
  ConfigurationEntryId,
  EnvironmentId,
  IntegrationId,
} from "../ids/index.js";

/**
 * Configuration metadata only. Secret values are never stored in plain `value` fields.
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
