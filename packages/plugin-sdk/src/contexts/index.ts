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

/**
 * Host-injected credential material for a single invocation.
 * Plugins must not persist, log, or return these values.
 */
export interface PluginInvocationCredentials {
  accessToken?: string;
}

export interface AuthenticateContext {
  pluginId: string;
  integrationId: string;
  /** Host-injected; never persist or echo back. */
  credentials?: PluginInvocationCredentials;
  connectionMetadata?: Record<string, unknown>;
}

export interface DiscoveryContext {
  pluginId: string;
  integrationId: string;
  projectId?: string;
  environmentId?: string;
  credentials?: PluginInvocationCredentials;
  connectionMetadata?: Record<string, unknown>;
}

export interface InspectContext {
  pluginId: string;
  integrationId: string;
  resource: ResourceBinding;
  credentials?: PluginInvocationCredentials;
  connectionMetadata?: Record<string, unknown>;
}

export interface PlanContext {
  pluginId: string;
  integrationId: string;
  resource: ResourceBinding;
  observed: ObservedResourceState;
  desired: DesiredResourceState;
  credentials?: PluginInvocationCredentials;
  connectionMetadata?: Record<string, unknown>;
}

export interface ApplyContext {
  pluginId: string;
  integrationId: string;
  resource: ResourceBinding;
  approvedPlan: ApprovedChangePlan;
  credentials?: PluginInvocationCredentials;
  connectionMetadata?: Record<string, unknown>;
}

export interface VerifyContext {
  pluginId: string;
  integrationId: string;
  resource: ResourceBinding;
  approvedPlan: ApprovedChangePlan;
  applyResult: ApplyResult;
  credentials?: PluginInvocationCredentials;
  connectionMetadata?: Record<string, unknown>;
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
