/**
 * Data-only setup contributions. Rayvan UI owns widgets; plugins declare steps.
 */

export type PluginSetupAuthMethod = "github_device_flow" | "pat";

export const PLUGIN_SETUP_AUTH_METHODS: readonly PluginSetupAuthMethod[] = [
  "github_device_flow",
  "pat",
] as const;

export type PluginSetupWidget =
  | "auth_method_select"
  | "device_flow"
  | "pat_input"
  | "confirm"
  | "info";

export const PLUGIN_SETUP_WIDGETS: readonly PluginSetupWidget[] = [
  "auth_method_select",
  "device_flow",
  "pat_input",
  "confirm",
  "info",
] as const;

export interface PluginSetupStepDefinition {
  id: string;
  title: string;
  description?: string;
  widget: PluginSetupWidget;
}

export interface PluginSetupContribution {
  authMethods: PluginSetupAuthMethod[];
  steps: PluginSetupStepDefinition[];
}

export interface PluginPublisherInfo {
  name: string;
  url?: string;
  supportUrl?: string;
}

export interface PluginEntrypointDefinition {
  /** OOP native launcher runtime (Node SEA or equivalent). */
  runtime: "native";
  /** Binary basename under package `bin/` (e.g. `rayvan-plugin-github`). */
  binary: string;
  protocolVersion: string;
}
