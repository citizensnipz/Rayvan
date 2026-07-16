import { RAYVAN_PLUGIN_API_VERSION } from "../api-version.js";
import { CAPABILITY_HANDLER_KEYS } from "../capabilities/index.js";
import type {
  ApplyResult,
  ApprovedChangePlan,
  AuthenticateResult,
  ChangeOperation,
  ChangePlan,
  DiscoveredResource,
  EvaluateFindingsResult,
  ObservedResourceState,
  PluginFindingDetection,
  PluginFindingEvidence,
  PluginFindingObservedState,
  PluginFindingRemediation,
  PluginFindingRuleDefinition,
  PluginResource,
  PluginSafeFindingValue,
  VerificationResult,
} from "../contracts/index.js";
import {
  isPluginFindingCategory,
  isPluginFindingSeverity,
  PLUGIN_FINDING_REMEDIATION_TYPES,
} from "../contracts/index.js";
import type {
  EvaluateFindingsContext,
  PluginFindingEnvironmentContext,
  PluginFindingObservedStateContext,
  PluginFindingResourceContext,
} from "../contexts/index.js";
import type { PluginExecutionActor } from "../execution/actor.js";
import {
  PluginValidationError,
  PluginVersionError,
} from "../errors/index.js";
import {
  PLUGIN_ACCENT_COLOR_PATTERN,
  PLUGIN_CAPABILITIES,
  PLUGIN_FOREGROUND_MODES,
  PLUGIN_PERMISSIONS,
  PLUGIN_THEME_SURFACES,
  type PluginCapability,
  type PluginManifest,
  type PluginPermission,
  type PluginPresentationDefinition,
  type PluginResourceTypeDefinition,
} from "../manifest/index.js";
import type { RayvanPlugin } from "../plugin.js";

const PLUGIN_ID_PATTERN = /^[a-z][a-z0-9-]*$/;
const SEMVER_PATTERN =
  /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;
const RESOURCE_TYPE_ID_PATTERN = /^[a-z][a-z0-9._-]*$/;

/** Keys treated as secret-bearing (aligned with execution/redaction.ts). */
const SENSITIVE_KEY_PATTERN =
  /^(token|secret|password|authorization|apiKey|accessToken|refreshToken)$/i;

/**
 * Obvious secret material in free-text evidence / remediation strings.
 * Rejected at validation time (host must not persist plaintext secrets).
 */
const OBVIOUS_SECRET_PATTERNS: readonly RegExp[] = [
  /\bBearer\s+[A-Za-z0-9\-._~+/]+=*/i,
  /\b(?:api[_-]?key|secret|password|token|access[_-]?token|refresh[_-]?token)\s*[:=]\s*\S+/i,
  /\bsk-(?:live|test)?[_-]?[A-Za-z0-9]{16,}\b/,
  /\bghp_[A-Za-z0-9]{20,}\b/,
  /\bgho_[A-Za-z0-9]{20,}\b/,
  /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/i,
];

const SAFE_VALUE_ACCESSES = new Set([
  "readable",
  "fingerprint",
  "masked",
  "locked",
  "name_only",
  "unknown",
]);

const EVIDENCE_TYPES = new Set([
  "configuration_comparison",
  "connection_error",
  "resource_state",
  "deployment_state",
  "message",
]);

const REMEDIATION_TYPES = new Set<string>(PLUGIN_FINDING_REMEDIATION_TYPES);

function assertNonEmptyString(
  value: unknown,
  field: string,
  pluginId?: string,
): asserts value is string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new PluginValidationError(`${field} must be a non-empty string`, {
      pluginId,
    });
  }
}

function assertSemver(value: string, field: string, pluginId?: string): void {
  if (!SEMVER_PATTERN.test(value)) {
    throw new PluginValidationError(
      `${field} must be a semantic version (got "${value}")`,
      { pluginId },
    );
  }
}

function assertUniqueStrings(
  values: readonly string[],
  field: string,
  pluginId?: string,
): void {
  const seen = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) {
      throw new PluginValidationError(`Duplicate ${field}: "${value}"`, {
        pluginId,
      });
    }
    seen.add(value);
  }
}

function isPluginCapability(value: unknown): value is PluginCapability {
  return (
    typeof value === "string" &&
    (PLUGIN_CAPABILITIES as readonly string[]).includes(value)
  );
}

function isPluginPermission(value: unknown): value is PluginPermission {
  return (
    typeof value === "string" &&
    (PLUGIN_PERMISSIONS as readonly string[]).includes(value)
  );
}

function assertNoObviousSecrets(
  value: string,
  field: string,
  pluginId?: string,
): void {
  for (const pattern of OBVIOUS_SECRET_PATTERNS) {
    if (pattern.test(value)) {
      throw new PluginValidationError(
        `${field} appears to contain a secret and must be redacted`,
        { pluginId },
      );
    }
  }
}

function assertNoSensitiveKeys(
  value: unknown,
  field: string,
  pluginId?: string,
  seen = new WeakSet<object>(),
): void {
  if (value === null || value === undefined) {
    return;
  }
  if (typeof value === "string") {
    assertNoObviousSecrets(value, field, pluginId);
    return;
  }
  if (typeof value === "function") {
    throw new PluginValidationError(
      `${field} must be serializable (functions are not allowed)`,
      { pluginId },
    );
  }
  if (Array.isArray(value)) {
    if (seen.has(value)) {
      return;
    }
    seen.add(value);
    for (let i = 0; i < value.length; i += 1) {
      assertNoSensitiveKeys(value[i], `${field}[${i}]`, pluginId, seen);
    }
    return;
  }
  if (typeof value === "object") {
    if (seen.has(value)) {
      return;
    }
    seen.add(value);
    for (const [key, nested] of Object.entries(
      value as Record<string, unknown>,
    )) {
      if (SENSITIVE_KEY_PATTERN.test(key)) {
        throw new PluginValidationError(
          `${field}.${key} must not include secret-bearing keys`,
          { pluginId },
        );
      }
      if (typeof nested === "function") {
        throw new PluginValidationError(
          `${field}.${key} must be serializable (functions are not allowed)`,
          { pluginId },
        );
      }
      assertNoSensitiveKeys(nested, `${field}.${key}`, pluginId, seen);
    }
  }
}

function assertNamespacedRuleId(
  ruleId: string,
  pluginId: string,
  field: string,
): void {
  assertNonEmptyString(ruleId, field, pluginId);
  const prefix = `${pluginId}.`;
  if (!ruleId.startsWith(prefix) || ruleId.length <= prefix.length) {
    throw new PluginValidationError(
      `${field} "${ruleId}" must be namespaced as "${pluginId}...."`,
      { pluginId },
    );
  }
}

function validateResourceTypeDefinition(
  resourceType: PluginResourceTypeDefinition,
  pluginId: string,
): void {
  assertNonEmptyString(resourceType.id, "resourceTypes[].id", pluginId);
  if (!RESOURCE_TYPE_ID_PATTERN.test(resourceType.id)) {
    throw new PluginValidationError(
      `resourceTypes[].id "${resourceType.id}" must match ${RESOURCE_TYPE_ID_PATTERN}`,
      { pluginId },
    );
  }
  assertNonEmptyString(resourceType.name, "resourceTypes[].name", pluginId);
  assertNonEmptyString(
    resourceType.schemaVersion,
    "resourceTypes[].schemaVersion",
    pluginId,
  );
  assertSemver(
    resourceType.schemaVersion,
    "resourceTypes[].schemaVersion",
    pluginId,
  );
  if (
    resourceType.description !== undefined &&
    typeof resourceType.description !== "string"
  ) {
    throw new PluginValidationError(
      "resourceTypes[].description must be a string when provided",
      { pluginId },
    );
  }
}

function validateFindingRuleDefinition(
  rule: PluginFindingRuleDefinition,
  pluginId: string,
  index: number,
): void {
  const field = `findingRules[${index}]`;
  assertNamespacedRuleId(rule.id, pluginId, `${field}.id`);
  assertNonEmptyString(rule.name, `${field}.name`, pluginId);
  assertNonEmptyString(rule.description, `${field}.description`, pluginId);
  if (!isPluginFindingCategory(rule.category)) {
    throw new PluginValidationError(
      `${field}.category has unsupported value "${String(rule.category)}"`,
      { pluginId },
    );
  }
  if (!isPluginFindingSeverity(rule.defaultSeverity)) {
    throw new PluginValidationError(
      `${field}.defaultSeverity has unsupported value "${String(rule.defaultSeverity)}"`,
      { pluginId },
    );
  }
  if (rule.documentationUrl !== undefined) {
    assertNonEmptyString(
      rule.documentationUrl,
      `${field}.documentationUrl`,
      pluginId,
    );
  }
}

export function validatePluginManifest(manifest: PluginManifest): void {
  assertNonEmptyString(manifest.id, "manifest.id");
  if (!PLUGIN_ID_PATTERN.test(manifest.id)) {
    throw new PluginValidationError(
      `manifest.id "${manifest.id}" must match ${PLUGIN_ID_PATTERN}`,
    );
  }

  const pluginId = manifest.id;

  assertNonEmptyString(manifest.name, "manifest.name", pluginId);
  assertNonEmptyString(manifest.version, "manifest.version", pluginId);
  assertSemver(manifest.version, "manifest.version", pluginId);
  assertNonEmptyString(manifest.publisher, "manifest.publisher", pluginId);
  assertNonEmptyString(
    manifest.rayvanApiVersion,
    "manifest.rayvanApiVersion",
    pluginId,
  );

  if (manifest.rayvanApiVersion !== RAYVAN_PLUGIN_API_VERSION) {
    throw new PluginVersionError(
      `Unsupported rayvanApiVersion "${manifest.rayvanApiVersion}" (expected "${RAYVAN_PLUGIN_API_VERSION}")`,
      { pluginId },
    );
  }

  if (manifest.minimumRayvanVersion !== undefined) {
    assertNonEmptyString(
      manifest.minimumRayvanVersion,
      "manifest.minimumRayvanVersion",
      pluginId,
    );
    assertSemver(
      manifest.minimumRayvanVersion,
      "manifest.minimumRayvanVersion",
      pluginId,
    );
  }

  if (manifest.description !== undefined && typeof manifest.description !== "string") {
    throw new PluginValidationError(
      "manifest.description must be a string when provided",
      { pluginId },
    );
  }

  if (!Array.isArray(manifest.capabilities)) {
    throw new PluginValidationError("manifest.capabilities must be an array", {
      pluginId,
    });
  }
  for (const capability of manifest.capabilities) {
    if (!isPluginCapability(capability)) {
      throw new PluginValidationError(
        `Unsupported capability: "${String(capability)}"`,
        { pluginId },
      );
    }
  }
  assertUniqueStrings(manifest.capabilities, "capability", pluginId);

  if (!Array.isArray(manifest.permissions)) {
    throw new PluginValidationError("manifest.permissions must be an array", {
      pluginId,
    });
  }
  for (const permission of manifest.permissions) {
    if (!isPluginPermission(permission)) {
      throw new PluginValidationError(
        `Unsupported permission: "${String(permission)}"`,
        { pluginId },
      );
    }
  }
  assertUniqueStrings(manifest.permissions, "permission", pluginId);

  if (!Array.isArray(manifest.resourceTypes)) {
    throw new PluginValidationError("manifest.resourceTypes must be an array", {
      pluginId,
    });
  }
  for (const resourceType of manifest.resourceTypes) {
    validateResourceTypeDefinition(resourceType, pluginId);
  }
  assertUniqueStrings(
    manifest.resourceTypes.map((resourceType) => resourceType.id),
    "resource type id",
    pluginId,
  );

  if (manifest.findingRules !== undefined) {
    if (!Array.isArray(manifest.findingRules)) {
      throw new PluginValidationError(
        "manifest.findingRules must be an array when provided",
        { pluginId },
      );
    }
    for (let i = 0; i < manifest.findingRules.length; i += 1) {
      validateFindingRuleDefinition(manifest.findingRules[i]!, pluginId, i);
    }
    assertUniqueStrings(
      manifest.findingRules.map((rule) => rule.id),
      "finding rule id",
      pluginId,
    );
  }

  if (manifest.presentation !== undefined) {
    validatePluginPresentation(manifest.presentation, pluginId);
  }
}

function validatePluginPresentation(
  presentation: PluginPresentationDefinition,
  pluginId: string,
): void {
  if (
    presentation.supportsMultipleConnections !== undefined &&
    typeof presentation.supportsMultipleConnections !== "boolean"
  ) {
    throw new PluginValidationError(
      "presentation.supportsMultipleConnections must be a boolean when provided",
      { pluginId },
    );
  }

  if (presentation.icon !== undefined) {
    const { icon } = presentation;
    if (typeof icon !== "object" || icon === null) {
      throw new PluginValidationError("presentation.icon must be an object", {
        pluginId,
      });
    }
    assertNonEmptyString(icon.label, "presentation.icon.label", pluginId);
    if (icon.iconId !== undefined) {
      assertNonEmptyString(icon.iconId, "presentation.icon.iconId", pluginId);
      if (!/^[a-z][a-z0-9-]*$/.test(icon.iconId)) {
        throw new PluginValidationError(
          `presentation.icon.iconId "${icon.iconId}" must be a lowercase kebab-case id`,
          { pluginId },
        );
      }
    }
    if (icon.initials !== undefined) {
      assertNonEmptyString(icon.initials, "presentation.icon.initials", pluginId);
      if (icon.initials.length > 3) {
        throw new PluginValidationError(
          "presentation.icon.initials must be at most 3 characters",
          { pluginId },
        );
      }
    }
  }

  if (presentation.theme !== undefined) {
    const { theme } = presentation;
    if (typeof theme !== "object" || theme === null) {
      throw new PluginValidationError("presentation.theme must be an object", {
        pluginId,
      });
    }
    if (
      !PLUGIN_THEME_SURFACES.includes(
        theme.surface as (typeof PLUGIN_THEME_SURFACES)[number],
      )
    ) {
      throw new PluginValidationError(
        `presentation.theme.surface must be one of: ${PLUGIN_THEME_SURFACES.join(", ")}`,
        { pluginId },
      );
    }
    if (theme.accentColor !== undefined) {
      assertNonEmptyString(
        theme.accentColor,
        "presentation.theme.accentColor",
        pluginId,
      );
      if (!PLUGIN_ACCENT_COLOR_PATTERN.test(theme.accentColor)) {
        throw new PluginValidationError(
          'presentation.theme.accentColor must be a #RRGGBB hex color',
          { pluginId },
        );
      }
    }
    if (
      theme.foregroundMode !== undefined &&
      !PLUGIN_FOREGROUND_MODES.includes(theme.foregroundMode)
    ) {
      throw new PluginValidationError(
        `presentation.theme.foregroundMode must be one of: ${PLUGIN_FOREGROUND_MODES.join(", ")}`,
        { pluginId },
      );
    }
  }
}

export function validatePluginHandlers(plugin: RayvanPlugin): void {
  const { manifest } = plugin;
  const pluginId = manifest.id;

  for (const capability of PLUGIN_CAPABILITIES) {
    const handlerKey = CAPABILITY_HANDLER_KEYS[capability];
    const handler = plugin[handlerKey];
    const declared = manifest.capabilities.includes(capability);
    const hasHandler = typeof handler === "function";

    if (declared && !hasHandler) {
      throw new PluginValidationError(
        `Capability "${capability}" is declared but handler "${handlerKey}" is missing`,
        { pluginId },
      );
    }

    if (hasHandler && !declared) {
      throw new PluginValidationError(
        `Handler "${handlerKey}" is present but capability "${capability}" is not declared`,
        { pluginId },
      );
    }
  }
}

export function validatePlugin(plugin: RayvanPlugin): void {
  validatePluginManifest(plugin.manifest);
  validatePluginHandlers(plugin);
}

export function validateDiscoveredResource(
  resource: DiscoveredResource,
  pluginId?: string,
): void {
  assertNonEmptyString(
    resource.providerResourceId,
    "discovered.providerResourceId",
    pluginId,
  );
  assertNonEmptyString(resource.resourceType, "discovered.resourceType", pluginId);
  assertNonEmptyString(resource.name, "discovered.name", pluginId);
  assertNonEmptyString(
    resource.schemaVersion,
    "discovered.schemaVersion",
    pluginId,
  );
  if (
    resource.metadata === null ||
    typeof resource.metadata !== "object" ||
    Array.isArray(resource.metadata)
  ) {
    throw new PluginValidationError("discovered.metadata must be an object", {
      pluginId,
    });
  }
}

export function validatePluginResource(
  resource: PluginResource,
): void {
  assertNonEmptyString(resource.id, "resource.id", resource.pluginId);
  assertNonEmptyString(resource.pluginId, "resource.pluginId");
  assertNonEmptyString(
    resource.providerResourceId,
    "resource.providerResourceId",
    resource.pluginId,
  );
  assertNonEmptyString(resource.resourceType, "resource.resourceType", resource.pluginId);
  assertNonEmptyString(resource.name, "resource.name", resource.pluginId);
  assertNonEmptyString(
    resource.pluginVersion,
    "resource.pluginVersion",
    resource.pluginId,
  );
  assertNonEmptyString(
    resource.schemaVersion,
    "resource.schemaVersion",
    resource.pluginId,
  );
  if (
    resource.metadata === null ||
    typeof resource.metadata !== "object" ||
    Array.isArray(resource.metadata)
  ) {
    throw new PluginValidationError("resource.metadata must be an object", {
      pluginId: resource.pluginId,
    });
  }
}

function validatePluginExecutionActor(
  actor: PluginExecutionActor,
  field: string,
  pluginId?: string,
): void {
  assertNonEmptyString(actor.id, `${field}.id`, pluginId);
  const allowedTypes = new Set(["user", "mcp_agent", "system"]);
  if (!allowedTypes.has(actor.type)) {
    throw new PluginValidationError(
      `${field}.type has unsupported value "${String(actor.type)}"`,
      { pluginId },
    );
  }
  if (
    actor.type !== "system" &&
    actor.displayName !== undefined &&
    typeof actor.displayName !== "string"
  ) {
    throw new PluginValidationError(
      `${field}.displayName must be a string when provided`,
      { pluginId },
    );
  }
}

export function validateApprovedChangePlan(
  approved: ApprovedChangePlan,
): void {
  const pluginId = approved.plan?.pluginId;
  validateChangePlan(approved.plan);
  assertNonEmptyString(approved.approvalId, "approvedPlan.approvalId", pluginId);
  assertNonEmptyString(approved.approvedAt, "approvedPlan.approvedAt", pluginId);

  if (!Array.isArray(approved.approvedOperationIds)) {
    throw new PluginValidationError(
      "approvedPlan.approvedOperationIds must be an array",
      { pluginId },
    );
  }
  for (const operationId of approved.approvedOperationIds) {
    assertNonEmptyString(
      operationId,
      "approvedPlan.approvedOperationIds[]",
      pluginId,
    );
  }
  assertUniqueStrings(
    approved.approvedOperationIds,
    "approved operation id",
    pluginId,
  );

  if (approved.approvedBy !== undefined) {
    validatePluginExecutionActor(
      approved.approvedBy,
      "approvedPlan.approvedBy",
      pluginId,
    );
  }

  if (
    approved.destructiveApproval !== undefined &&
    typeof approved.destructiveApproval !== "boolean"
  ) {
    throw new PluginValidationError(
      "approvedPlan.destructiveApproval must be a boolean when provided",
      { pluginId },
    );
  }
}

export function validateAuthenticateResult(
  result: AuthenticateResult,
  pluginId?: string,
): void {
  if (typeof result.ok !== "boolean") {
    throw new PluginValidationError("authenticateResult.ok must be a boolean", {
      pluginId,
    });
  }
  assertNonEmptyString(result.message, "authenticateResult.message", pluginId);
}

export function validateChangePlan(plan: ChangePlan): void {
  assertNonEmptyString(plan.id, "changePlan.id", plan.pluginId);
  assertNonEmptyString(plan.pluginId, "changePlan.pluginId");
  assertNonEmptyString(plan.resourceId, "changePlan.resourceId", plan.pluginId);
  assertNonEmptyString(plan.summary, "changePlan.summary", plan.pluginId);

  if (!Array.isArray(plan.operations)) {
    throw new PluginValidationError("changePlan.operations must be an array", {
      pluginId: plan.pluginId,
    });
  }
  if (!Array.isArray(plan.warnings)) {
    throw new PluginValidationError("changePlan.warnings must be an array", {
      pluginId: plan.pluginId,
    });
  }
  if (typeof plan.destructive !== "boolean") {
    throw new PluginValidationError("changePlan.destructive must be a boolean", {
      pluginId: plan.pluginId,
    });
  }

  for (const operation of plan.operations) {
    validateChangeOperation(operation, plan.pluginId);
  }
  assertUniqueStrings(
    plan.operations.map((operation) => operation.id),
    "change operation id",
    plan.pluginId,
  );
}

function validateChangeOperation(
  operation: ChangeOperation,
  pluginId: string,
): void {
  assertNonEmptyString(operation.id, "changeOperation.id", pluginId);
  assertNonEmptyString(operation.type, "changeOperation.type", pluginId);
  assertNonEmptyString(
    operation.description,
    "changeOperation.description",
    pluginId,
  );
  if (typeof operation.requiresApproval !== "boolean") {
    throw new PluginValidationError(
      "changeOperation.requiresApproval must be a boolean",
      { pluginId },
    );
  }
}

const OBSERVED_STATUSES = new Set([
  "ready",
  "degraded",
  "unavailable",
  "unknown",
]);

export function validateObservedResourceState(
  state: ObservedResourceState,
): void {
  assertNonEmptyString(state.resourceId, "observed.resourceId", state.pluginId);
  assertNonEmptyString(state.pluginId, "observed.pluginId");
  assertNonEmptyString(
    state.resourceType,
    "observed.resourceType",
    state.pluginId,
  );
  assertNonEmptyString(state.observedAt, "observed.observedAt", state.pluginId);
  if (!OBSERVED_STATUSES.has(state.status)) {
    throw new PluginValidationError(
      `observed.status has unsupported value "${String(state.status)}"`,
      { pluginId: state.pluginId },
    );
  }
  if (
    state.attributes === null ||
    typeof state.attributes !== "object" ||
    Array.isArray(state.attributes)
  ) {
    throw new PluginValidationError("observed.attributes must be an object", {
      pluginId: state.pluginId,
    });
  }
}

export function validateApplyResult(
  result: ApplyResult,
  pluginId?: string,
): void {
  if (typeof result.ok !== "boolean") {
    throw new PluginValidationError("applyResult.ok must be a boolean", {
      pluginId,
    });
  }
  assertNonEmptyString(result.message, "applyResult.message", pluginId);
  if (!Array.isArray(result.appliedOperationIds)) {
    throw new PluginValidationError(
      "applyResult.appliedOperationIds must be an array",
      { pluginId },
    );
  }
  if (result.resultingState !== undefined) {
    validateObservedResourceState(result.resultingState);
  }
}

export function validateVerificationResult(
  result: VerificationResult,
  pluginId?: string,
): void {
  if (typeof result.ok !== "boolean") {
    throw new PluginValidationError("verificationResult.ok must be a boolean", {
      pluginId,
    });
  }
  assertNonEmptyString(result.message, "verificationResult.message", pluginId);
  if (result.observed !== undefined) {
    validateObservedResourceState(result.observed);
  }
  if (result.mismatches !== undefined && !Array.isArray(result.mismatches)) {
    throw new PluginValidationError(
      "verificationResult.mismatches must be an array when provided",
      { pluginId },
    );
  }
}

function validateSafeFindingValue(
  value: PluginSafeFindingValue,
  field: string,
  pluginId: string,
): void {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new PluginValidationError(`${field} must be an object`, { pluginId });
  }
  if (!SAFE_VALUE_ACCESSES.has(value.access)) {
    throw new PluginValidationError(
      `${field}.access has unsupported value "${String(value.access)}"`,
      { pluginId },
    );
  }
  if (typeof value.sensitive !== "boolean") {
    throw new PluginValidationError(`${field}.sensitive must be a boolean`, {
      pluginId,
    });
  }
  if (value.access === "readable") {
    if (value.sensitive !== false) {
      throw new PluginValidationError(
        `${field}.sensitive must be false for readable values`,
        { pluginId },
      );
    }
    assertNonEmptyString(value.value, `${field}.value`, pluginId);
    assertNoObviousSecrets(value.value, `${field}.value`, pluginId);
  } else if (value.access === "fingerprint") {
    if (value.sensitive !== true) {
      throw new PluginValidationError(
        `${field}.sensitive must be true for fingerprint values`,
        { pluginId },
      );
    }
    assertNonEmptyString(value.fingerprint, `${field}.fingerprint`, pluginId);
  } else if (value.access === "masked") {
    if (value.sensitive !== true) {
      throw new PluginValidationError(
        `${field}.sensitive must be true for masked values`,
        { pluginId },
      );
    }
    if (value.maskedValue !== undefined) {
      assertNonEmptyString(value.maskedValue, `${field}.maskedValue`, pluginId);
      assertNoObviousSecrets(value.maskedValue, `${field}.maskedValue`, pluginId);
    }
  }
}

function validateFindingObservedState(
  state: PluginFindingObservedState,
  field: string,
  pluginId: string,
): void {
  if (state === null || typeof state !== "object" || Array.isArray(state)) {
    throw new PluginValidationError(`${field} must be an object`, { pluginId });
  }
  validateSafeFindingValue(state.value, `${field}.value`, pluginId);
  if (state.label !== undefined) {
    assertNonEmptyString(state.label, `${field}.label`, pluginId);
  }
  if (state.inSync !== undefined && typeof state.inSync !== "boolean") {
    throw new PluginValidationError(
      `${field}.inSync must be a boolean when provided`,
      { pluginId },
    );
  }
}

function validateFindingEvidence(
  evidence: PluginFindingEvidence,
  field: string,
  pluginId: string,
): void {
  if (evidence === null || typeof evidence !== "object" || Array.isArray(evidence)) {
    throw new PluginValidationError(`${field} must be an object`, { pluginId });
  }
  if (!EVIDENCE_TYPES.has(evidence.type)) {
    throw new PluginValidationError(
      `${field}.type has unsupported value "${String(evidence.type)}"`,
      { pluginId },
    );
  }

  switch (evidence.type) {
    case "configuration_comparison":
      assertNonEmptyString(
        evidence.configurationKeyId,
        `${field}.configurationKeyId`,
        pluginId,
      );
      assertNonEmptyString(
        evidence.environmentId,
        `${field}.environmentId`,
        pluginId,
      );
      if (evidence.expectedState !== undefined) {
        validateSafeFindingValue(
          evidence.expectedState,
          `${field}.expectedState`,
          pluginId,
        );
      }
      if (!Array.isArray(evidence.observedStates)) {
        throw new PluginValidationError(
          `${field}.observedStates must be an array`,
          { pluginId },
        );
      }
      for (let i = 0; i < evidence.observedStates.length; i += 1) {
        validateFindingObservedState(
          evidence.observedStates[i]!,
          `${field}.observedStates[${i}]`,
          pluginId,
        );
      }
      break;
    case "connection_error":
      assertNonEmptyString(evidence.connectionId, `${field}.connectionId`, pluginId);
      assertNonEmptyString(evidence.safeMessage, `${field}.safeMessage`, pluginId);
      assertNoObviousSecrets(evidence.safeMessage, `${field}.safeMessage`, pluginId);
      if (evidence.errorCode !== undefined) {
        assertNonEmptyString(evidence.errorCode, `${field}.errorCode`, pluginId);
      }
      break;
    case "resource_state":
      assertNonEmptyString(
        evidence.resourceBindingId,
        `${field}.resourceBindingId`,
        pluginId,
      );
      assertNonEmptyString(evidence.state, `${field}.state`, pluginId);
      assertNoObviousSecrets(evidence.state, `${field}.state`, pluginId);
      break;
    case "deployment_state":
      assertNonEmptyString(evidence.status, `${field}.status`, pluginId);
      assertNoObviousSecrets(evidence.status, `${field}.status`, pluginId);
      if (evidence.deploymentId !== undefined) {
        assertNonEmptyString(
          evidence.deploymentId,
          `${field}.deploymentId`,
          pluginId,
        );
      }
      break;
    case "message":
      assertNonEmptyString(evidence.message, `${field}.message`, pluginId);
      assertNoObviousSecrets(evidence.message, `${field}.message`, pluginId);
      break;
    default: {
      const _exhaustive: never = evidence;
      void _exhaustive;
      throw new PluginValidationError(`${field}.type is unsupported`, {
        pluginId,
      });
    }
  }

  assertNoSensitiveKeys(evidence, field, pluginId);
}

function validateFindingRemediation(
  remediation: PluginFindingRemediation,
  field: string,
  pluginId: string,
): void {
  if (
    remediation === null ||
    typeof remediation !== "object" ||
    Array.isArray(remediation)
  ) {
    throw new PluginValidationError(`${field} must be an object`, { pluginId });
  }
  if (typeof (remediation as { run?: unknown }).run === "function") {
    throw new PluginValidationError(
      `${field} must not include executable callbacks`,
      { pluginId },
    );
  }
  if (!REMEDIATION_TYPES.has(remediation.type)) {
    throw new PluginValidationError(
      `${field}.type has unsupported value "${String(remediation.type)}"`,
      { pluginId },
    );
  }

  assertNonEmptyString(remediation.label, `${field}.label`, pluginId);
  assertNoObviousSecrets(remediation.label, `${field}.label`, pluginId);

  if (remediation.type === "manual") {
    assertNonEmptyString(
      remediation.instructions,
      `${field}.instructions`,
      pluginId,
    );
    assertNoObviousSecrets(
      remediation.instructions,
      `${field}.instructions`,
      pluginId,
    );
  }

  assertNoSensitiveKeys(remediation, field, pluginId);
}

export function validatePluginFindingDetection(
  detection: PluginFindingDetection,
  pluginId: string,
): void {
  assertNamespacedRuleId(detection.ruleId, pluginId, "detection.ruleId");
  assertNonEmptyString(detection.title, "detection.title", pluginId);
  assertNonEmptyString(detection.summary, "detection.summary", pluginId);
  assertNoObviousSecrets(detection.title, "detection.title", pluginId);
  assertNoObviousSecrets(detection.summary, "detection.summary", pluginId);

  if (detection.description !== undefined) {
    assertNonEmptyString(detection.description, "detection.description", pluginId);
    assertNoObviousSecrets(
      detection.description,
      "detection.description",
      pluginId,
    );
  }

  if (
    detection.severity !== undefined &&
    !isPluginFindingSeverity(detection.severity)
  ) {
    throw new PluginValidationError(
      `detection.severity has unsupported value "${String(detection.severity)}"`,
      { pluginId },
    );
  }

  if (
    detection.scope === null ||
    typeof detection.scope !== "object" ||
    Array.isArray(detection.scope)
  ) {
    throw new PluginValidationError("detection.scope must be an object", {
      pluginId,
    });
  }

  if (!Array.isArray(detection.evidence)) {
    throw new PluginValidationError("detection.evidence must be an array", {
      pluginId,
    });
  }
  for (let i = 0; i < detection.evidence.length; i += 1) {
    validateFindingEvidence(
      detection.evidence[i]!,
      `detection.evidence[${i}]`,
      pluginId,
    );
  }

  if (detection.remediation !== undefined) {
    validateFindingRemediation(
      detection.remediation,
      "detection.remediation",
      pluginId,
    );
  }

  if (!Array.isArray(detection.fingerprintParts)) {
    throw new PluginValidationError(
      "detection.fingerprintParts must be an array",
      { pluginId },
    );
  }
  if (detection.fingerprintParts.length === 0) {
    throw new PluginValidationError(
      "detection.fingerprintParts must not be empty",
      { pluginId },
    );
  }
  for (let i = 0; i < detection.fingerprintParts.length; i += 1) {
    assertNonEmptyString(
      detection.fingerprintParts[i],
      `detection.fingerprintParts[${i}]`,
      pluginId,
    );
  }

  if (detection.metadata !== undefined) {
    if (
      detection.metadata === null ||
      typeof detection.metadata !== "object" ||
      Array.isArray(detection.metadata)
    ) {
      throw new PluginValidationError(
        "detection.metadata must be an object when provided",
        { pluginId },
      );
    }
    assertNoSensitiveKeys(detection.metadata, "detection.metadata", pluginId);
  }
}

function validateFindingEnvironmentContext(
  environment: PluginFindingEnvironmentContext,
  field: string,
  pluginId: string,
): void {
  assertNonEmptyString(environment.id, `${field}.id`, pluginId);
  if (environment.name !== undefined) {
    assertNonEmptyString(environment.name, `${field}.name`, pluginId);
  }
}

function validateFindingResourceContext(
  resource: PluginFindingResourceContext,
  field: string,
  pluginId: string,
): void {
  if (resource === null || typeof resource !== "object" || Array.isArray(resource)) {
    throw new PluginValidationError(`${field} must be an object`, { pluginId });
  }
  if (
    resource.resourceBindingId === undefined &&
    resource.discoveredResourceId === undefined
  ) {
    throw new PluginValidationError(
      `${field} must include resourceBindingId or discoveredResourceId`,
      { pluginId },
    );
  }
}

function validateFindingObservedStateContext(
  state: PluginFindingObservedStateContext,
  field: string,
  pluginId: string,
): void {
  validateSafeFindingValue(state.value, `${field}.value`, pluginId);
}

export function validateEvaluateFindingsContext(
  context: EvaluateFindingsContext,
): void {
  const pluginId = context.pluginId;
  assertNonEmptyString(context.pluginId, "evaluateFindings.pluginId");
  assertNonEmptyString(context.projectId, "evaluateFindings.projectId", pluginId);
  assertNonEmptyString(
    context.connectionId,
    "evaluateFindings.connectionId",
    pluginId,
  );

  if (context.integrationId !== undefined) {
    assertNonEmptyString(
      context.integrationId,
      "evaluateFindings.integrationId",
      pluginId,
    );
  }
  if (context.lastEvaluatedAt !== undefined) {
    assertNonEmptyString(
      context.lastEvaluatedAt,
      "evaluateFindings.lastEvaluatedAt",
      pluginId,
    );
  }

  if (!Array.isArray(context.environments)) {
    throw new PluginValidationError(
      "evaluateFindings.environments must be an array",
      { pluginId },
    );
  }
  for (let i = 0; i < context.environments.length; i += 1) {
    validateFindingEnvironmentContext(
      context.environments[i]!,
      `evaluateFindings.environments[${i}]`,
      pluginId,
    );
  }

  if (!Array.isArray(context.resources)) {
    throw new PluginValidationError(
      "evaluateFindings.resources must be an array",
      { pluginId },
    );
  }
  for (let i = 0; i < context.resources.length; i += 1) {
    validateFindingResourceContext(
      context.resources[i]!,
      `evaluateFindings.resources[${i}]`,
      pluginId,
    );
  }

  if (!Array.isArray(context.observedStates)) {
    throw new PluginValidationError(
      "evaluateFindings.observedStates must be an array",
      { pluginId },
    );
  }
  for (let i = 0; i < context.observedStates.length; i += 1) {
    validateFindingObservedStateContext(
      context.observedStates[i]!,
      `evaluateFindings.observedStates[${i}]`,
      pluginId,
    );
  }
}

export function validateEvaluateFindingsResult(
  result: EvaluateFindingsResult,
  pluginId: string,
): void {
  if (result === null || typeof result !== "object" || Array.isArray(result)) {
    throw new PluginValidationError(
      "evaluateFindingsResult must be an object",
      { pluginId },
    );
  }
  if (!Array.isArray(result.detections)) {
    throw new PluginValidationError(
      "evaluateFindingsResult.detections must be an array",
      { pluginId },
    );
  }
  if (!Array.isArray(result.warnings)) {
    throw new PluginValidationError(
      "evaluateFindingsResult.warnings must be an array",
      { pluginId },
    );
  }
  for (const warning of result.warnings) {
    assertNonEmptyString(
      warning,
      "evaluateFindingsResult.warnings[]",
      pluginId,
    );
    assertNoObviousSecrets(
      warning,
      "evaluateFindingsResult.warnings[]",
      pluginId,
    );
  }
  for (const detection of result.detections) {
    validatePluginFindingDetection(detection, pluginId);
  }
}
