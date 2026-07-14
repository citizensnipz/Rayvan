import { RAYVAN_PLUGIN_API_VERSION } from "../api-version.js";
import { CAPABILITY_HANDLER_KEYS } from "../capabilities/index.js";
import type {
  ApplyResult,
  ApprovedChangePlan,
  AuthenticateResult,
  ChangeOperation,
  ChangePlan,
  DiscoveredResource,
  ObservedResourceState,
  PluginResource,
  VerificationResult,
} from "../contracts/index.js";
import type { PluginExecutionActor } from "../execution/actor.js";
import {
  PluginValidationError,
  PluginVersionError,
} from "../errors/index.js";
import {
  PLUGIN_CAPABILITIES,
  PLUGIN_PERMISSIONS,
  type PluginCapability,
  type PluginManifest,
  type PluginPermission,
  type PluginResourceTypeDefinition,
} from "../manifest/index.js";
import type { RayvanPlugin } from "../plugin.js";

const PLUGIN_ID_PATTERN = /^[a-z][a-z0-9-]*$/;
const SEMVER_PATTERN =
  /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;
const RESOURCE_TYPE_ID_PATTERN = /^[a-z][a-z0-9._-]*$/;

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
