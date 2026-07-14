import type { PluginManifest } from "./manifest/index.js";
import type {
  PluginActionContext,
  PluginConfigurationContext,
  PluginConnectionContext,
  PluginDiscoveryContext,
  PluginExecutionContext,
  PluginHealthContext,
  PluginInitializeContext,
} from "./contexts/index.js";
import type {
  ActionExecutionResult,
  ActionRequest,
  ApprovedPluginActionPlan,
  ConfigurationSnapshot,
  ConnectionTestResult,
  DiscoveredResource,
  HealthSnapshot,
  PluginActionPlan,
} from "./contracts/index.js";

export interface RayvanPlugin {
  manifest: PluginManifest;

  initialize(context: PluginInitializeContext): Promise<void>;

  testConnection(context: PluginConnectionContext): Promise<ConnectionTestResult>;

  discoverResources(context: PluginDiscoveryContext): Promise<DiscoveredResource[]>;

  collectConfiguration(
    context: PluginConfigurationContext,
  ): Promise<ConfigurationSnapshot>;

  collectHealth(context: PluginHealthContext): Promise<HealthSnapshot>;

  planAction(
    context: PluginActionContext,
    request: ActionRequest,
  ): Promise<PluginActionPlan>;

  executeAction(
    context: PluginExecutionContext,
    plan: ApprovedPluginActionPlan,
  ): Promise<ActionExecutionResult>;

  dispose(): Promise<void>;
}

export class NotImplementedPluginError extends Error {
  constructor(phase: string) {
    super(`Plugin capability not implemented: ${phase}`);
    this.name = "NotImplementedPluginError";
  }
}

export * from "./manifest/index.js";
export * from "./capabilities/index.js";
export * from "./contexts/index.js";
export * from "./contracts/index.js";
export * from "./protocol/index.js";
export * from "./validation/index.js";
