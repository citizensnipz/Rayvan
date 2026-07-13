export type PluginCapability =
  | "resource-discovery"
  | "configuration-read"
  | "health-read"
  | "action-plan"
  | "action-execute";

export interface PluginManifest {
  id: string;
  name: string;
  version: string;
  protocolVersion: string;
  capabilities: readonly PluginCapability[];
}
