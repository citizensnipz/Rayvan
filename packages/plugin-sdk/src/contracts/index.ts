import type { PluginExecutionActor } from "../execution/actor.js";

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
 * Apply must receive this envelope — never a raw unapproved ChangePlan.
 */
export interface ApprovedChangePlan {
  plan: ChangePlan;
  approvalId: string;
  approvedAt: string;
  /** Operation ids explicitly approved by the host. Required. */
  approvedOperationIds: string[];
  approvedBy?: PluginExecutionActor;
  /** Must be true when the plan or any operation is destructive. */
  destructiveApproval?: boolean;
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

/**
 * Finding category taxonomy — mirrors `@rayvan/core` FindingCategory.
 * Kept local so plugin-sdk does not depend on core.
 */
export type PluginFindingCategory =
  | "configuration"
  | "environment"
  | "integration"
  | "resource"
  | "deployment"
  | "security"
  | "availability"
  | "drift"
  | "permission"
  | "mapping"
  | "other";

export type PluginFindingSeverity = "info" | "warning" | "error" | "critical";

export const PLUGIN_FINDING_CATEGORIES: readonly PluginFindingCategory[] = [
  "configuration",
  "environment",
  "integration",
  "resource",
  "deployment",
  "security",
  "availability",
  "drift",
  "permission",
  "mapping",
  "other",
] as const;

export const PLUGIN_FINDING_SEVERITIES: readonly PluginFindingSeverity[] = [
  "info",
  "warning",
  "error",
  "critical",
] as const;

export function isPluginFindingCategory(
  value: string,
): value is PluginFindingCategory {
  return (PLUGIN_FINDING_CATEGORIES as readonly string[]).includes(value);
}

export function isPluginFindingSeverity(
  value: string,
): value is PluginFindingSeverity {
  return (PLUGIN_FINDING_SEVERITIES as readonly string[]).includes(value);
}

/** Safe value representation — never plaintext secrets. Mirrors core SafeFindingValue. */
export type PluginSafeFindingValue =
  | {
      access: "readable";
      value: string;
      sensitive: false;
    }
  | {
      access: "fingerprint";
      fingerprint: string;
      sensitive: true;
    }
  | {
      access: "masked";
      maskedValue?: string;
      sensitive: true;
    }
  | {
      access: "locked" | "name_only" | "unknown";
      sensitive: boolean;
    };

export interface PluginFindingObservedState {
  pluginId?: string;
  connectionId?: string;
  discoveredResourceId?: string;
  resourceBindingId?: string;
  label?: string;
  value: PluginSafeFindingValue;
  inSync?: boolean;
  observedAt?: string;
}

/** Serializable evidence — no secrets, no functions. Mirrors core FindingEvidence. */
export type PluginFindingEvidence =
  | {
      type: "configuration_comparison";
      configurationKeyId: string;
      environmentId: string;
      expectedState?: PluginSafeFindingValue;
      observedStates: PluginFindingObservedState[];
    }
  | {
      type: "connection_error";
      connectionId: string;
      errorCode?: string;
      safeMessage: string;
    }
  | {
      type: "resource_state";
      resourceBindingId: string;
      state: string;
      observedAt?: string;
    }
  | {
      type: "deployment_state";
      deploymentId?: string;
      status: string;
      observedAt?: string;
    }
  | {
      type: "message";
      message: string;
    };

/** Serializable remediation descriptors — never executable callbacks. */
export type PluginFindingRemediation =
  | {
      type: "generate_change_plan";
      pluginId?: string;
      connectionId?: string;
      resourceBindingId?: string;
      configurationKeyIds?: string[];
      label: string;
    }
  | {
      type: "open_environment";
      environmentId: string;
      label: string;
    }
  | {
      type: "open_integration";
      integrationId?: string;
      connectionId?: string;
      label: string;
    }
  | {
      type: "open_resource";
      resourceBindingId: string;
      label: string;
    }
  | {
      type: "reauthenticate";
      connectionId: string;
      label: string;
    }
  | {
      type: "resync";
      pluginId?: string;
      connectionId?: string;
      label: string;
    }
  | {
      type: "manual";
      label: string;
      instructions: string;
    };

export const PLUGIN_FINDING_REMEDIATION_TYPES = [
  "generate_change_plan",
  "open_environment",
  "open_integration",
  "open_resource",
  "reauthenticate",
  "resync",
  "manual",
] as const;

export type PluginFindingRemediationType =
  (typeof PLUGIN_FINDING_REMEDIATION_TYPES)[number];

/**
 * Declared finding rule for a plugin manifest.
 * Rule ids must be namespaced as `${pluginId}....`.
 */
export interface PluginFindingRuleDefinition {
  id: string;
  name: string;
  description: string;
  category: PluginFindingCategory;
  defaultSeverity: PluginFindingSeverity;
  documentationUrl?: string;
}

export interface PluginFindingDetectionScope {
  environmentId?: string;
  resourceBindingId?: string;
  discoveredResourceId?: string;
  configurationKeyId?: string;
  deploymentId?: string;
}

/**
 * Detection emitted by `evaluate_findings`.
 * Plugins must not assign Finding record IDs or mutate Finding history.
 */
export interface PluginFindingDetection {
  ruleId: string;
  severity?: PluginFindingSeverity;
  title: string;
  summary: string;
  description?: string;
  scope: PluginFindingDetectionScope;
  evidence: PluginFindingEvidence[];
  remediation?: PluginFindingRemediation;
  /** Parts used to build a stable fingerprint (not titles/timestamps). */
  fingerprintParts: string[];
  metadata?: Record<string, unknown>;
}

export interface EvaluateFindingsResult {
  detections: PluginFindingDetection[];
  warnings: string[];
}
