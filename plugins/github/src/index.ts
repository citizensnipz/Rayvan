import { randomUUID } from "node:crypto";

import type {
  ApplyContext,
  ApplyResult,
  AuthenticateContext,
  ChangePlan,
  DesiredResourceState,
  DiscoveredResource,
  DiscoverHandler,
  EvaluateFindingsContext,
  EvaluateFindingsResult,
  InspectContext,
  ObservedResourceState,
  PlanContext,
  PluginFindingDetection,
  RayvanPlugin,
  VerificationResult,
  VerifyContext,
} from "@rayvan/plugin-sdk";

import {
  GITHUB_ACTIONS_VARIABLES_RESOURCE_TYPE,
  GITHUB_ACTIONS_VARIABLES_SCHEMA_VERSION,
  GITHUB_PLUGIN_ID,
  GITHUB_REPOSITORY_RESOURCE_TYPE,
  GITHUB_REPOSITORY_SCHEMA_VERSION,
} from "./constants.js";
import { createClientFromContext } from "./context.js";
import { resetGithubFixtureStore } from "./fixture-store.js";
import { manifest } from "./manifest.js";

export { GITHUB_PLUGIN_ID, GITHUB_PLUGIN_BINARY } from "./constants.js";
export { manifest };
export { resetGithubFixtureStore };
export { scanWorkflowReferences } from "./workflows.js";

function variablesProviderId(fullName: string): string {
  return `${fullName}#actions-variables`;
}

function parseVariablesProviderId(providerResourceId: string): string | null {
  const suffix = "#actions-variables";
  if (!providerResourceId.endsWith(suffix)) return null;
  return providerResourceId.slice(0, -suffix.length);
}

/**
 * Normalize Actions variables from persistence-safe shapes:
 * - preferred: `[{ name, value }, ...]` (avoids secret-like map keys in SQLite)
 * - legacy: `{ NAME: value }` object map
 */
function asVariableMap(value: unknown): Record<string, string> {
  if (!value) return {};
  if (Array.isArray(value)) {
    const result: Record<string, string> = {};
    for (const item of value) {
      if (!item || typeof item !== "object") continue;
      const record = item as Record<string, unknown>;
      if (typeof record.name === "string" && typeof record.value === "string") {
        result[record.name] = record.value;
      }
    }
    return result;
  }
  if (typeof value !== "object") return {};
  const result: Record<string, string> = {};
  for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
    if (typeof nested === "string") {
      result[key] = nested;
    }
  }
  return result;
}

function variablesAttribute(
  variables: ReadonlyArray<{ name: string; value: string }>,
): Array<{ name: string; value: string }> {
  return variables.map((item) => ({ name: item.name, value: item.value }));
}

async function authenticate(context: AuthenticateContext) {
  const client = createClientFromContext(context);
  const result = await client.authenticate();
  return {
    ok: result.ok,
    message: result.message,
  };
}

const discover: DiscoverHandler = async (context) => {
  const client = createClientFromContext(context);
  const repos = await client.listRepositories();
  const resources: DiscoveredResource[] = [];

  for (const repo of repos) {
    resources.push({
      providerResourceId: repo.fullName,
      resourceType: GITHUB_REPOSITORY_RESOURCE_TYPE,
      name: repo.fullName,
      metadata: {
        owner: repo.owner,
        private: repo.private,
        defaultBranch: repo.defaultBranch,
      },
      schemaVersion: GITHUB_REPOSITORY_SCHEMA_VERSION,
    });
    resources.push({
      providerResourceId: variablesProviderId(repo.fullName),
      resourceType: GITHUB_ACTIONS_VARIABLES_RESOURCE_TYPE,
      name: `${repo.fullName} Actions variables`,
      metadata: {
        repository: repo.fullName,
        owner: repo.owner,
        parentProviderResourceId: repo.fullName,
      },
      schemaVersion: GITHUB_ACTIONS_VARIABLES_SCHEMA_VERSION,
    });
  }

  return resources;
};

async function inspect(context: InspectContext): Promise<ObservedResourceState> {
  const client = createClientFromContext(context);
  const { resource } = context;

  if (resource.resourceType === GITHUB_REPOSITORY_RESOURCE_TYPE) {
    return {
      resourceId: resource.resourceId,
      pluginId: GITHUB_PLUGIN_ID,
      resourceType: resource.resourceType,
      observedAt: new Date().toISOString(),
      status: "ready",
      attributes: {
        fullName: resource.providerResourceId,
      },
    };
  }

  if (resource.resourceType !== GITHUB_ACTIONS_VARIABLES_RESOURCE_TYPE) {
    return {
      resourceId: resource.resourceId,
      pluginId: GITHUB_PLUGIN_ID,
      resourceType: resource.resourceType,
      observedAt: new Date().toISOString(),
      status: "unknown",
      attributes: {},
    };
  }

  const fullName =
    parseVariablesProviderId(resource.providerResourceId) ??
    String(resource.providerResourceId);
  const variables = await client.listActionsVariables(fullName);
  const workflowRefs = await client.scanWorkflows(fullName);

  return {
    resourceId: resource.resourceId,
    pluginId: GITHUB_PLUGIN_ID,
    resourceType: GITHUB_ACTIONS_VARIABLES_RESOURCE_TYPE,
    observedAt: new Date().toISOString(),
    status: "ready",
    attributes: {
      repository: fullName,
      // Array form keeps variable *names* out of object keys so SQLite secret
      // scanners do not false-positive on names like API_TOKEN.
      variables: variablesAttribute(variables),
      /** Secrets are presence-only when discovered via workflow refs. */
      secrets: workflowRefs.secretNames.map((name) => ({
        name,
        access: "name_only" as const,
        sensitive: true as const,
      })),
      workflowRefs,
      configurationOccurrences: [
        ...variables.map((item) => ({
          key: item.name,
          kind: "actions_variable",
          access: "readable",
          value: item.value,
        })),
        ...workflowRefs.secretNames.map((name) => ({
          key: name,
          kind: "actions_secret",
          access: "name_only",
        })),
      ],
    },
  };
}

async function plan(context: PlanContext): Promise<ChangePlan> {
  const observedVars = asVariableMap(context.observed.attributes.variables);
  const desiredVars = asVariableMap(context.desired.attributes.variables);
  const operations: ChangePlan["operations"] = [];

  for (const [name, value] of Object.entries(desiredVars)) {
    if (observedVars[name] === value) continue;
    const exists = Object.prototype.hasOwnProperty.call(observedVars, name);
    operations.push({
      id: randomUUID(),
      type: exists ? "github.actions_variable.update" : "github.actions_variable.create",
      description: exists
        ? `Update Actions variable ${name}`
        : `Create Actions variable ${name}`,
      requiresApproval: true,
      path: name,
      before: exists ? observedVars[name] : undefined,
      after: value,
    });
  }

  return {
    id: randomUUID(),
    pluginId: GITHUB_PLUGIN_ID,
    resourceId: context.resource.resourceId,
    summary:
      operations.length === 0
        ? "Actions variables already match desired state"
        : `Apply ${operations.length} Actions variable change(s)`,
    operations,
    warnings: [],
    destructive: false,
  };
}

async function apply(context: ApplyContext): Promise<ApplyResult> {
  const client = createClientFromContext(context);
  const fullName =
    parseVariablesProviderId(context.resource.providerResourceId) ??
    String(context.resource.providerResourceId);
  const appliedOperationIds: string[] = [];

  for (const operation of context.approvedPlan.plan.operations) {
    if (!context.approvedPlan.approvedOperationIds.includes(operation.id)) {
      continue;
    }
    const name = String(operation.path ?? "");
    const value = String(operation.after ?? "");
    if (!name) continue;
    await client.upsertActionsVariable(fullName, name, value);
    appliedOperationIds.push(operation.id);
  }

  const variables = await client.listActionsVariables(fullName);
  const resultingState: ObservedResourceState = {
    resourceId: context.resource.resourceId,
    pluginId: GITHUB_PLUGIN_ID,
    resourceType: GITHUB_ACTIONS_VARIABLES_RESOURCE_TYPE,
    observedAt: new Date().toISOString(),
    status: "ready",
    attributes: {
      repository: fullName,
      variables: variablesAttribute(variables),
    },
  };

  return {
    ok: true,
    message: `Applied ${appliedOperationIds.length} Actions variable operation(s)`,
    appliedOperationIds,
    resultingState,
  };
}

async function verify(context: VerifyContext): Promise<VerificationResult> {
  const client = createClientFromContext(context);
  const fullName =
    parseVariablesProviderId(context.resource.providerResourceId) ??
    String(context.resource.providerResourceId);
  const variables = await client.listActionsVariables(fullName);
  const observedMap = asVariableMap(variablesAttribute(variables));

  const desiredFromPlan = asVariableMap(
    Object.fromEntries(
      context.approvedPlan.plan.operations
        .filter((op) =>
          context.approvedPlan.approvedOperationIds.includes(op.id),
        )
        .map((op) => [String(op.path ?? ""), String(op.after ?? "")]),
    ),
  );

  const mismatches: string[] = [];
  for (const [name, value] of Object.entries(desiredFromPlan)) {
    if (observedMap[name] !== value) {
      mismatches.push(name);
    }
  }

  const observed: ObservedResourceState = {
    resourceId: context.resource.resourceId,
    pluginId: GITHUB_PLUGIN_ID,
    resourceType: GITHUB_ACTIONS_VARIABLES_RESOURCE_TYPE,
    observedAt: new Date().toISOString(),
    status: mismatches.length === 0 ? "ready" : "degraded",
    attributes: {
      repository: fullName,
      variables: variablesAttribute(variables),
    },
  };

  return {
    ok: mismatches.length === 0,
    message:
      mismatches.length === 0
        ? "Actions variables match approved plan"
        : `Variables out of sync: ${mismatches.join(", ")}`,
    observed,
    mismatches,
  };
}

async function evaluateFindings(
  context: EvaluateFindingsContext,
): Promise<EvaluateFindingsResult> {
  const detections: PluginFindingDetection[] = [];
  const warnings: string[] = [];

  for (const resource of context.resources) {
    if (
      resource.resourceType &&
      resource.resourceType !== GITHUB_ACTIONS_VARIABLES_RESOURCE_TYPE
    ) {
      continue;
    }
    const observed = context.observedStates.find(
      (state) =>
        state.resourceBindingId === resource.resourceBindingId ||
        state.discoveredResourceId === resource.discoveredResourceId,
    );
    // Host may pass structured attributes via observed state labels / metadata-like values.
    // For fixture/unit tests we also accept readable JSON in value.value when access=readable.
    let variables: Record<string, string> = {};
    let referenced: string[] = [];
    if (observed?.value.access === "readable") {
      try {
        const parsed = JSON.parse(observed.value.value) as {
          variables?: unknown;
          workflowVariableRefs?: string[];
          workflowRefs?: { variableNames?: string[] };
        };
        variables = asVariableMap(parsed.variables);
        referenced =
          parsed.workflowVariableRefs ??
          parsed.workflowRefs?.variableNames ??
          [];
      } catch {
        warnings.push("Could not parse observed finding payload for GitHub resource");
      }
    }

    const defined = new Set(Object.keys(variables));
    const refs = new Set(referenced);

    for (const name of defined) {
      if (!refs.has(name)) {
        detections.push({
          ruleId: `${GITHUB_PLUGIN_ID}.unused-actions-variable`,
          severity: "warning",
          title: `Unused Actions variable ${name}`,
          summary: `Variable ${name} is defined but not referenced by discovered workflows.`,
          scope: {
            environmentId: resource.environmentId,
            resourceBindingId: resource.resourceBindingId,
          },
          evidence: [
            {
              type: "message",
              message: `Variable ${name} has no vars.${name} references in scanned workflows.`,
            },
          ],
          fingerprintParts: [
            GITHUB_PLUGIN_ID,
            "unused-actions-variable",
            resource.providerResourceId ?? "unknown",
            name,
          ],
        });
      }
    }

    for (const name of refs) {
      if (!defined.has(name)) {
        detections.push({
          ruleId: `${GITHUB_PLUGIN_ID}.missing-referenced-variable`,
          severity: "error",
          title: `Missing Actions variable ${name}`,
          summary: `Workflows reference vars.${name} but the variable is not defined.`,
          scope: {
            environmentId: resource.environmentId,
            resourceBindingId: resource.resourceBindingId,
          },
          evidence: [
            {
              type: "message",
              message: `vars.${name} is referenced but missing from repository Actions variables.`,
            },
          ],
          fingerprintParts: [
            GITHUB_PLUGIN_ID,
            "missing-referenced-variable",
            resource.providerResourceId ?? "unknown",
            name,
          ],
        });
      }
    }
  }

  return { detections, warnings };
}

/** Desired-state helper for hosts/tests building Actions variable plans. */
export function desiredActionsVariables(
  resourceId: string,
  variables: Record<string, string>,
): DesiredResourceState {
  return {
    resourceId,
    pluginId: GITHUB_PLUGIN_ID,
    resourceType: GITHUB_ACTIONS_VARIABLES_RESOURCE_TYPE,
    attributes: {
      variables: Object.entries(variables).map(([name, value]) => ({
        name,
        value,
      })),
    },
  };
}

export const plugin: RayvanPlugin = {
  manifest,
  authenticate,
  discover,
  inspect,
  plan,
  apply,
  verify,
  evaluateFindings,
};

export default plugin;
