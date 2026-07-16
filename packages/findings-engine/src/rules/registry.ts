import type { FindingRuleDefinition } from "@rayvan/core";

import { validateRuleDefinition } from "../validation.js";

/** Stable core rule IDs — do not rename once shipped. */
export const CORE_FINDING_RULE_IDS = {
  CONFIGURATION_MISSING_REQUIRED: "rayvan.configuration.missing-required",
  CONFIGURATION_MISMATCH: "rayvan.configuration.mismatch",
  CONFIGURATION_UNAPPLIED: "rayvan.configuration.unapplied",
  CONFIGURATION_REMOTE_CHANGED: "rayvan.configuration.remote-changed",
  CONFIGURATION_PARTIALLY_APPLIED: "rayvan.configuration.partially-applied",
  CONFIGURATION_COMPARISON_UNAVAILABLE:
    "rayvan.configuration.comparison-unavailable",
  CONFIGURATION_UNMANAGED: "rayvan.configuration.unmanaged",
  CONFIGURATION_INSPECTION_STALE: "rayvan.configuration.inspection-stale",
  RESOURCE_UNMAPPED: "rayvan.resource.unmapped",
  RESOURCE_MISSING: "rayvan.resource.missing",
  ENVIRONMENT_NO_RESOURCES: "rayvan.environment.no-resources",
  ENVIRONMENT_PENDING_MAPPING_SUGGESTION:
    "rayvan.environment.pending-mapping-suggestion",
  INTEGRATION_CONNECTION_EXPIRED: "rayvan.integration.connection-expired",
  INTEGRATION_CONNECTION_REVOKED: "rayvan.integration.connection-revoked",
  INTEGRATION_AUTHENTICATION_FAILED:
    "rayvan.integration.authentication-failed",
  INTEGRATION_PERMISSION_MISSING: "rayvan.integration.permission-missing",
  INTEGRATION_PLUGIN_DISABLED: "rayvan.integration.plugin-disabled",
  INTEGRATION_PLUGIN_INCOMPATIBLE: "rayvan.integration.plugin-incompatible",
  INTEGRATION_SYNC_FAILURE: "rayvan.integration.sync-failure",
  INTEGRATION_CREDENTIAL_MISSING: "rayvan.integration.credential-missing",
  CHANGE_APPLY_FAILED: "rayvan.change.apply-failed",
  CHANGE_APPLY_PARTIAL: "rayvan.change.apply-partial",
  CHANGE_VERIFICATION_FAILED: "rayvan.change.verification-failed",
  CHANGE_PLAN_STALE: "rayvan.change.plan-stale",
} as const;

export type CoreFindingRuleId =
  (typeof CORE_FINDING_RULE_IDS)[keyof typeof CORE_FINDING_RULE_IDS];

const RAYVAN_SOURCE = { type: "rayvan" as const };

function rule(
  partial: Omit<FindingRuleDefinition, "source" | "enabledByDefault"> & {
    enabledByDefault?: boolean;
  },
): FindingRuleDefinition {
  return validateRuleDefinition({
    ...partial,
    source: RAYVAN_SOURCE,
    enabledByDefault: partial.enabledByDefault ?? true,
  });
}

export const CORE_FINDING_RULES: readonly FindingRuleDefinition[] = [
  rule({
    id: CORE_FINDING_RULE_IDS.CONFIGURATION_MISSING_REQUIRED,
    name: "Missing required configuration",
    description:
      "A required configuration key has no remote value in an environment.",
    category: "configuration",
    defaultSeverity: "error",
    supportedObjectTypes: ["configuration_key", "environment"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
    name: "Configuration mismatch",
    description:
      "Desired configuration disagrees with observed values on one or more resources.",
    category: "drift",
    defaultSeverity: "warning",
    supportedObjectTypes: ["configuration_key", "environment"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.CONFIGURATION_UNAPPLIED,
    name: "Unapplied configuration changes",
    description:
      "Saved desired configuration differs from last-applied state (not editor drafts).",
    category: "configuration",
    defaultSeverity: "warning",
    supportedObjectTypes: ["configuration_key", "environment"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.CONFIGURATION_REMOTE_CHANGED,
    name: "Remote configuration changed",
    description:
      "Observed remote values changed while desired still matches last-applied.",
    category: "drift",
    defaultSeverity: "warning",
    supportedObjectTypes: ["configuration_key", "environment"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.CONFIGURATION_PARTIALLY_APPLIED,
    name: "Partially applied configuration",
    description:
      "Some bound resources match desired configuration while others do not.",
    category: "configuration",
    defaultSeverity: "warning",
    supportedObjectTypes: ["configuration_key", "environment"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.CONFIGURATION_COMPARISON_UNAVAILABLE,
    name: "Configuration comparison unavailable",
    description:
      "Values are locked or otherwise inaccessible, so Rayvan cannot compare them.",
    category: "configuration",
    defaultSeverity: "info",
    supportedObjectTypes: ["configuration_key", "environment"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.CONFIGURATION_UNMANAGED,
    name: "Unmanaged configuration",
    description:
      "Observed configuration exists without a saved desired value in Rayvan.",
    category: "configuration",
    defaultSeverity: "info",
    supportedObjectTypes: ["configuration_key", "environment"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.CONFIGURATION_INSPECTION_STALE,
    name: "Stale configuration inspection",
    description:
      "Observed configuration has not been inspected recently (default threshold: 7 days).",
    category: "configuration",
    defaultSeverity: "info",
    supportedObjectTypes: ["configuration_key", "environment"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.RESOURCE_UNMAPPED,
    name: "Unmapped resource",
    description:
      "A discovered active resource has no environment binding.",
    category: "resource",
    defaultSeverity: "warning",
    supportedObjectTypes: ["resource", "environment"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.RESOURCE_MISSING,
    name: "Missing bound resource",
    description:
      "An active binding points at a resource that is missing or inaccessible.",
    category: "resource",
    defaultSeverity: "error",
    supportedObjectTypes: ["resource", "environment"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.ENVIRONMENT_NO_RESOURCES,
    name: "Environment has no resources",
    description:
      "An environment has zero active resource bindings (skips local_only and newly created).",
    category: "environment",
    defaultSeverity: "info",
    supportedObjectTypes: ["environment"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.ENVIRONMENT_PENDING_MAPPING_SUGGESTION,
    name: "Pending mapping suggestion",
    description: "A resource-to-environment mapping suggestion is pending review.",
    category: "mapping",
    defaultSeverity: "info",
    supportedObjectTypes: ["environment", "resource"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
    name: "Connection expired",
    description: "An integration connection credential or session has expired.",
    category: "integration",
    defaultSeverity: "error",
    supportedObjectTypes: ["connection", "integration"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_REVOKED,
    name: "Connection revoked",
    description: "An integration connection has been revoked.",
    category: "integration",
    defaultSeverity: "error",
    supportedObjectTypes: ["connection", "integration"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.INTEGRATION_AUTHENTICATION_FAILED,
    name: "Authentication failed",
    description: "An integration connection is in an authentication error state.",
    category: "integration",
    defaultSeverity: "error",
    supportedObjectTypes: ["connection", "integration"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.INTEGRATION_PERMISSION_MISSING,
    name: "Permission missing",
    description:
      "An integration reported a permission error (error code permission-related).",
    category: "permission",
    defaultSeverity: "warning",
    supportedObjectTypes: ["connection", "integration"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.INTEGRATION_PLUGIN_DISABLED,
    name: "Plugin disabled",
    description: "A connection references a disabled installed plugin.",
    category: "integration",
    defaultSeverity: "warning",
    supportedObjectTypes: ["connection", "integration"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.INTEGRATION_PLUGIN_INCOMPATIBLE,
    name: "Plugin incompatible",
    description: "A connection references an incompatible installed plugin.",
    category: "integration",
    defaultSeverity: "error",
    supportedObjectTypes: ["connection", "integration"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.INTEGRATION_SYNC_FAILURE,
    name: "Sync failure",
    description:
      "A connection has a failed sync more recent than its last successful sync.",
    category: "integration",
    defaultSeverity: "warning",
    supportedObjectTypes: ["connection", "integration"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.INTEGRATION_CREDENTIAL_MISSING,
    name: "Credential missing",
    description: "A connection has no credential reference.",
    category: "integration",
    defaultSeverity: "error",
    supportedObjectTypes: ["connection", "integration"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.CHANGE_APPLY_FAILED,
    name: "Change apply failed",
    description: "A change plan apply completed with failure.",
    category: "deployment",
    defaultSeverity: "error",
    supportedObjectTypes: ["change_plan", "resource"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.CHANGE_APPLY_PARTIAL,
    name: "Change apply partial",
    description: "A change plan apply completed only partially.",
    category: "deployment",
    defaultSeverity: "warning",
    supportedObjectTypes: ["change_plan", "resource"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.CHANGE_VERIFICATION_FAILED,
    name: "Change verification failed",
    description: "Post-apply verification reported failure.",
    category: "deployment",
    defaultSeverity: "error",
    supportedObjectTypes: ["change_plan", "resource"],
  }),
  rule({
    id: CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE,
    name: "Change plan stale",
    description:
      "Observed resource state changed since the change plan was created.",
    category: "deployment",
    defaultSeverity: "warning",
    supportedObjectTypes: ["change_plan", "resource"],
  }),
] as const;

const RULE_BY_ID = new Map(
  CORE_FINDING_RULES.map((definition) => [definition.id, definition]),
);

export function getCoreFindingRule(
  ruleId: string,
): FindingRuleDefinition | undefined {
  return RULE_BY_ID.get(ruleId);
}

export function listCoreFindingRules(): readonly FindingRuleDefinition[] {
  return CORE_FINDING_RULES;
}

export function listCoreFindingRuleIds(): readonly string[] {
  return CORE_FINDING_RULES.map((definition) => definition.id);
}
