import type { ConfigurationKeyId, ProjectId } from "../ids/index.js";

export type ConfigurationValueType =
  | "string"
  | "number"
  | "boolean"
  | "url"
  | "json"
  | "secret"
  | "unknown";

export type ConfigurationKeySource = "discovered" | "manual" | "imported";

/**
 * Project-scoped logical configuration key identity.
 * One key may have many occurrences across environments and providers.
 */
export interface ConfigurationKey {
  id: ConfigurationKeyId;
  projectId: ProjectId;
  name: string;
  description?: string;
  valueType: ConfigurationValueType;
  required: boolean;
  sensitive: boolean;
  source: ConfigurationKeySource;
  createdAt: string;
  updatedAt: string;
}
