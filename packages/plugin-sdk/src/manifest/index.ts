export type PluginCapability =
  | "authenticate"
  | "discover"
  | "inspect"
  | "plan"
  | "apply"
  | "verify"
  | "evaluate_findings";

export const PLUGIN_CAPABILITIES: readonly PluginCapability[] = [
  "authenticate",
  "discover",
  "inspect",
  "plan",
  "apply",
  "verify",
  "evaluate_findings",
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

export type {
  PluginForegroundMode,
  PluginIconDefinition,
  PluginPresentationDefinition,
  PluginThemeDefinition,
  PluginThemeSurface,
} from "./presentation.js";
export {
  PLUGIN_ACCENT_COLOR_PATTERN,
  PLUGIN_FOREGROUND_MODES,
  PLUGIN_THEME_SURFACES,
} from "./presentation.js";
export type {
  PluginEntrypointDefinition,
  PluginPublisherInfo,
  PluginSetupAuthMethod,
  PluginSetupContribution,
  PluginSetupStepDefinition,
  PluginSetupWidget,
} from "./setup.js";
export {
  PLUGIN_SETUP_AUTH_METHODS,
  PLUGIN_SETUP_WIDGETS,
} from "./setup.js";

import type { PluginFindingRuleDefinition } from "../contracts/index.js";
import type { PluginPresentationDefinition } from "./presentation.js";
import type {
  PluginEntrypointDefinition,
  PluginPublisherInfo,
  PluginSetupContribution,
} from "./setup.js";

/**
 * Versioned plugin identity and capability declaration.
 * Stable string IDs are suitable for storage and serialization.
 *
 * Plugin ids may be kebab-case (`example-local`) or reverse-DNS
 * (`io.rayvan.github`).
 */
export interface PluginManifest {
  id: string;
  name: string;
  description?: string;
  version: string;
  /** Stable publisher string (keep for storage/compat). */
  publisher: string;

  rayvanApiVersion: string;
  minimumRayvanVersion?: string;

  capabilities: PluginCapability[];
  permissions: PluginPermission[];

  resourceTypes: PluginResourceTypeDefinition[];

  /**
   * Optional finding rule declarations for `evaluate_findings`.
   * Rule ids must be namespaced as `${pluginId}....`.
   */
  findingRules?: PluginFindingRuleDefinition[];

  /** Optional host UI presentation (icons/theme). Serializable only. */
  presentation?: PluginPresentationDefinition;

  /** Optional richer publisher metadata (display / support). */
  publisherInfo?: PluginPublisherInfo;

  /** Optional OOP entrypoints declared for packaged plugins. */
  entrypoints?: PluginEntrypointDefinition[];

  /** Optional allowlisted network hosts for host policy display. */
  networkHosts?: string[];

  /** Optional data-only setup contribution (UI owns widgets). */
  setup?: PluginSetupContribution;
}
