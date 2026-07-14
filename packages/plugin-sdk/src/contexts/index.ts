import type {
  ApplyResult,
  ApprovedChangePlan,
  DesiredResourceState,
  ObservedResourceState,
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
