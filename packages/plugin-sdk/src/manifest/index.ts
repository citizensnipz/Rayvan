export type PluginCapability =
  | "authenticate"
  | "discover"
  | "inspect"
  | "plan"
  | "apply"
  | "verify";

export const PLUGIN_CAPABILITIES: readonly PluginCapability[] = [
  "authenticate",
  "discover",
  "inspect",
  "plan",
  "apply",
  "verify",
] as const;

export type PluginPermission =
  | "network"
  | "read_secrets"
  | "write_remote_configuration"
  | "read_local_files"
  | "write_local_files";

export const PLUGIN_PERMISSIONS: readonly PluginPermission[] = [
  "network",
  "read_secrets",
  "write_remote_configuration",
  "read_local_files",
  "write_local_files",
] as const;

export interface PluginResourceTypeDefinition {
  id: string;
  name: string;
  description?: string;
  schemaVersion: string;
}

/**
 * Versioned plugin identity and capability declaration.
 * Stable string IDs are suitable for storage and serialization.
 */
export interface PluginManifest {
  id: string;
  name: string;
  description?: string;
  version: string;
  publisher: string;

  rayvanApiVersion: string;
  minimumRayvanVersion?: string;

  capabilities: PluginCapability[];
  permissions: PluginPermission[];

  resourceTypes: PluginResourceTypeDefinition[];
}
