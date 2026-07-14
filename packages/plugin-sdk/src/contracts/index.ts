/**
 * Generic plugin resource envelope. Provider-specific schemas stay inside plugins.
 */
export interface PluginResource {
  id: string;
  pluginId: string;
  providerResourceId: string;
  resourceType: string;
  name: string;

  projectId?: string;
  environmentId?: string;

  metadata: Record<string, unknown>;

  pluginVersion: string;
  schemaVersion: string;
}

/** Resource discovered before it is bound into a Rayvan project. */
export interface DiscoveredResource {
  providerResourceId: string;
  resourceType: string;
  name: string;
  metadata: Record<string, unknown>;
  schemaVersion: string;
}

/** Binding between a Rayvan resource id and a provider-native resource. */
export interface ResourceBinding {
  resourceId: string;
  pluginId: string;
  providerResourceId: string;
  resourceType: string;
  projectId?: string;
  environmentId?: string;
}

export type ObservedResourceStatus =
  | "ready"
  | "degraded"
  | "unavailable"
  | "unknown";

export interface ObservedCheck {
  name: string;
  status: "pass" | "warn" | "fail";
  message?: string;
}

export interface ObservedResourceState {
  resourceId: string;
  pluginId: string;
  resourceType: string;
  observedAt: string;
  status: ObservedResourceStatus;
  attributes: Record<string, unknown>;
  checks?: ObservedCheck[];
}

export interface DesiredResourceState {
  resourceId: string;
  pluginId: string;
  resourceType: string;
  attributes: Record<string, unknown>;
}

/**
 * Descriptive, serializable mutation step. Must not contain executable functions.
 */
export interface ChangeOperation {
  id: string;
  type: string;
  description: string;
  path?: string;
  before?: unknown;
  after?: unknown;
  destructive?: boolean;
  requiresApproval: boolean;
}

export interface ChangePlan {
  id: string;
  pluginId: string;
  resourceId: string;
  summary: string;
  operations: ChangeOperation[];
  warnings: string[];
  destructive: boolean;
}

/**
 * Host-approved plan reference. Approval creation and persistence belong to Core.
 */
export interface ApprovedChangePlan {
  plan: ChangePlan;
  approvalId: string;
  approvedAt: string;
}

export interface AuthenticateResult {
  ok: boolean;
  message: string;
}

export interface ApplyResult {
  ok: boolean;
  message: string;
  appliedOperationIds: string[];
  resultingState?: ObservedResourceState;
}

export interface VerificationResult {
  ok: boolean;
  message: string;
  observed?: ObservedResourceState;
  mismatches?: string[];
}
