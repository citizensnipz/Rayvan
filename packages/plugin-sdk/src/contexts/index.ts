import type {
  ApplyResult,
  ApprovedChangePlan,
  DesiredResourceState,
  ObservedResourceState,
  PluginSafeFindingValue,
  ResourceBinding,
} from "../contracts/index.js";

/**
 * Plugin contexts are plain serializable data.
 * Do not pass database clients, UI state, loggers, or service containers.
 */

export interface AuthenticateContext {
  pluginId: string;
  integrationId: string;
}

export interface DiscoveryContext {
  pluginId: string;
  integrationId: string;
  projectId?: string;
  environmentId?: string;
}

export interface InspectContext {
  pluginId: string;
  integrationId: string;
  resource: ResourceBinding;
}

export interface PlanContext {
  pluginId: string;
  integrationId: string;
  resource: ResourceBinding;
  observed: ObservedResourceState;
  desired: DesiredResourceState;
}

export interface ApplyContext {
  pluginId: string;
  integrationId: string;
  resource: ResourceBinding;
  approvedPlan: ApprovedChangePlan;
}

export interface VerifyContext {
  pluginId: string;
  integrationId: string;
  resource: ResourceBinding;
  approvedPlan: ApprovedChangePlan;
  applyResult: ApplyResult;
}

/** Minimal environment snapshot for findings evaluation. */
export interface PluginFindingEnvironmentContext {
  id: string;
  name?: string;
  kind?: string;
}

/** Minimal resource snapshot for findings evaluation. */
export interface PluginFindingResourceContext {
  resourceBindingId?: string;
  discoveredResourceId?: string;
  resourceType?: string;
  name?: string;
  environmentId?: string;
  providerResourceId?: string;
}

/** Minimal observed-state snapshot for findings evaluation. */
export interface PluginFindingObservedStateContext {
  resourceBindingId?: string;
  discoveredResourceId?: string;
  configurationKeyId?: string;
  environmentId?: string;
  label?: string;
  value: PluginSafeFindingValue;
  inSync?: boolean;
  observedAt?: string;
}

/**
 * Host-provided evaluation input for `evaluate_findings`.
 * Plugins return detections only; Finding persistence stays with the host.
 */
export interface EvaluateFindingsContext {
  pluginId: string;
  projectId: string;
  connectionId: string;
  integrationId?: string;
  environments: PluginFindingEnvironmentContext[];
  resources: PluginFindingResourceContext[];
  observedStates: PluginFindingObservedStateContext[];
  lastEvaluatedAt?: string;
}
