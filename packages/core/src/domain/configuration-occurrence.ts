import type {
  ConfigurationKeyId,
  ConfigurationOccurrenceId,
  EnvironmentId,
  ProjectId,
} from "../ids/index.js";

/**
 * How much of a discovered value Rayvan may surface.
 * Never infer that two locked/name_only values match.
 */
export type ConfigurationValueAccess =
  | "readable"
  | "masked"
  | "locked"
  | "name_only"
  | "missing";

/**
 * A single discovered (or recorded) occurrence of a configuration key
 * on a provider resource / connection within a project.
 */
export interface ConfigurationOccurrence {
  id: ConfigurationOccurrenceId;
  configurationKeyId: ConfigurationKeyId;
  projectId: ProjectId;
  environmentId?: EnvironmentId;

  pluginId: string;
  connectionId: string;
  discoveredResourceId: string;
  resourceBindingId?: string;

  /** Provider-facing key name (may differ from logical key name). */
  providerKey: string;

  valueAccess: ConfigurationValueAccess;

  /**
   * Safe visible value for non-sensitive readable occurrences only.
   * Sensitive plaintext must never be stored here — use secretValueRef.
   */
  observedValue?: string;
  maskedValue?: string;
  valueFingerprint?: string;
  /** Reference into credential / secure-secret storage when needed. */
  secretValueRef?: string;

  scope?: string;

  firstObservedAt: string;
  lastObservedAt: string;
}
