import type { FindingDetection, FindingEvaluationScope } from "@rayvan/core";

import { CORE_FINDING_RULE_IDS } from "../rules/registry.js";
import type { ProjectFindingsContext } from "../types.js";
import type { FindingEvaluator, FindingEvaluatorResult } from "./types.js";

const NEW_ENVIRONMENT_GRACE_MS = 24 * 60 * 60 * 1000;

function filterContextForScope(
  context: ProjectFindingsContext,
  scope: FindingEvaluationScope,
): ProjectFindingsContext {
  if (scope.type === "project") {
    return context;
  }
  if (scope.type === "environment") {
    return {
      ...context,
      environments: context.environments.filter(
        (environment) => environment.id === scope.environmentId,
      ),
      resourceBindings: context.resourceBindings.filter(
        (binding) => binding.environmentId === scope.environmentId,
      ),
      mappingSuggestions: context.mappingSuggestions.filter(
        (suggestion) =>
          suggestion.suggestedEnvironmentId === scope.environmentId ||
          suggestion.status === "pending",
      ),
    };
  }
  // connection scope: resources/bindings for that connection
  return {
    ...context,
    discoveredResources: context.discoveredResources.filter(
      (resource) => resource.connectionId === scope.connectionId,
    ),
    resourceBindings: context.resourceBindings.filter(
      (binding) => binding.connectionId === scope.connectionId,
    ),
    mappingSuggestions: context.mappingSuggestions.filter(
      (suggestion) => suggestion.connectionId === scope.connectionId,
    ),
  };
}

export function createEnvironmentEvaluator(): FindingEvaluator {
  const ruleIds = [
    CORE_FINDING_RULE_IDS.RESOURCE_UNMAPPED,
    CORE_FINDING_RULE_IDS.RESOURCE_MISSING,
    CORE_FINDING_RULE_IDS.ENVIRONMENT_NO_RESOURCES,
    CORE_FINDING_RULE_IDS.ENVIRONMENT_PENDING_MAPPING_SUGGESTION,
  ] as const;

  return {
    id: "rayvan.environment",
    ruleIds,
    evaluate({ projectId, scope, context, now }): FindingEvaluatorResult {
      const scoped = filterContextForScope(context, scope);
      const detections: FindingDetection[] = [];
      const evaluatedRuleIds: string[] = [];
      const nowMs = Date.parse(now);

      const activeBindingsByResource = new Map<string, typeof scoped.resourceBindings>();
      for (const binding of scoped.resourceBindings) {
        if (binding.bindingStatus !== "active") {
          continue;
        }
        const list = activeBindingsByResource.get(binding.discoveredResourceId) ?? [];
        list.push(binding);
        activeBindingsByResource.set(binding.discoveredResourceId, list);
      }

      // Unmapped: discovered resource attached/active without environment binding
      if (scope.type !== "environment") {
        evaluatedRuleIds.push(CORE_FINDING_RULE_IDS.RESOURCE_UNMAPPED);
        for (const resource of scoped.discoveredResources) {
          if (resource.discoveryStatus !== "active") {
            continue;
          }
          const bindings = activeBindingsByResource.get(resource.id) ?? [];
          const hasEnvironmentBinding = bindings.some(
            (binding) => Boolean(binding.environmentId),
          );
          if (hasEnvironmentBinding) {
            continue;
          }
          // Also treat "no active binding at all" as unmapped
          detections.push({
            ruleId: CORE_FINDING_RULE_IDS.RESOURCE_UNMAPPED,
            projectId,
            severity: "warning",
            title: "Unmapped resource",
            summary: `${resource.name} is discovered but not bound to an environment.`,
            scope: {
              connectionId: resource.connectionId,
              discoveredResourceId: resource.id,
            },
            evidence: [
              {
                type: "message",
                message: `Resource ${resource.id} (${resource.resourceType}) has no active environment binding.`,
              },
            ],
            fingerprintParts: [
              CORE_FINDING_RULE_IDS.RESOURCE_UNMAPPED,
              resource.id,
            ],
            remediation: {
              type: "manual",
              label: "Map resource to an environment",
              instructions:
                "Bind this discovered resource to a Rayvan environment.",
            },
          });
        }
      }

      // Missing: binding active but resource discoveryStatus missing/inaccessible
      evaluatedRuleIds.push(CORE_FINDING_RULE_IDS.RESOURCE_MISSING);
      const resourceById = new Map(
        scoped.discoveredResources.map((resource) => [resource.id, resource]),
      );
      for (const binding of scoped.resourceBindings) {
        if (binding.bindingStatus !== "active") {
          continue;
        }
        const resource = resourceById.get(binding.discoveredResourceId);
        if (
          !resource ||
          resource.discoveryStatus === "missing" ||
          resource.discoveryStatus === "inaccessible"
        ) {
          detections.push({
            ruleId: CORE_FINDING_RULE_IDS.RESOURCE_MISSING,
            projectId,
            severity: "error",
            title: "Missing bound resource",
            summary: `${binding.displayName ?? binding.discoveredResourceId} is bound but missing or inaccessible.`,
            scope: {
              environmentId: binding.environmentId,
              connectionId: binding.connectionId,
              discoveredResourceId: binding.discoveredResourceId,
              resourceBindingId: binding.id,
            },
            evidence: [
              {
                type: "resource_state",
                resourceBindingId: binding.id,
                state: resource?.discoveryStatus ?? "missing",
              },
            ],
            fingerprintParts: [
              CORE_FINDING_RULE_IDS.RESOURCE_MISSING,
              binding.id,
            ],
            remediation: {
              type: "open_resource",
              resourceBindingId: binding.id,
              label: "Open resource binding",
            },
          });
        }
      }

      // No resources: environment with zero bindings — skip local_only and <24h
      if (scope.type !== "connection") {
        evaluatedRuleIds.push(CORE_FINDING_RULE_IDS.ENVIRONMENT_NO_RESOURCES);
        for (const environment of scoped.environments) {
          if (environment.status === "local_only") {
            continue;
          }
          const createdMs = Date.parse(environment.createdAt);
          if (
            !Number.isNaN(createdMs) &&
            !Number.isNaN(nowMs) &&
            nowMs - createdMs < NEW_ENVIRONMENT_GRACE_MS
          ) {
            continue;
          }
          const bindingCount = scoped.resourceBindings.filter(
            (binding) =>
              binding.environmentId === environment.id &&
              binding.bindingStatus === "active",
          ).length;
          if (bindingCount > 0) {
            continue;
          }
          detections.push({
            ruleId: CORE_FINDING_RULE_IDS.ENVIRONMENT_NO_RESOURCES,
            projectId,
            severity: "info",
            title: "Environment has no resources",
            summary: `${environment.name} has no active resource bindings.`,
            scope: { environmentId: environment.id },
            evidence: [
              {
                type: "message",
                message: `Environment ${environment.id} has zero active bindings.`,
              },
            ],
            fingerprintParts: [
              CORE_FINDING_RULE_IDS.ENVIRONMENT_NO_RESOURCES,
              environment.id,
            ],
            remediation: {
              type: "open_environment",
              environmentId: environment.id,
              label: `Open ${environment.name}`,
            },
          });
        }
      }

      // Pending mapping suggestion — one finding per suggestion
      evaluatedRuleIds.push(
        CORE_FINDING_RULE_IDS.ENVIRONMENT_PENDING_MAPPING_SUGGESTION,
      );
      for (const suggestion of scoped.mappingSuggestions) {
        if (suggestion.status !== "pending") {
          continue;
        }
        if (
          scope.type === "environment" &&
          suggestion.suggestedEnvironmentId &&
          suggestion.suggestedEnvironmentId !== scope.environmentId
        ) {
          continue;
        }
        detections.push({
          ruleId: CORE_FINDING_RULE_IDS.ENVIRONMENT_PENDING_MAPPING_SUGGESTION,
          projectId,
          severity: "info",
          title: "Pending mapping suggestion",
          summary: suggestion.suggestedEnvironmentName
            ? `Suggested mapping to ${suggestion.suggestedEnvironmentName}.`
            : "A resource mapping suggestion is pending review.",
          scope: {
            environmentId: suggestion.suggestedEnvironmentId,
            connectionId: suggestion.connectionId,
            discoveredResourceId: suggestion.discoveredResourceId,
          },
          evidence: [
            {
              type: "message",
              message: `Mapping suggestion ${suggestion.id} is pending.`,
            },
          ],
          fingerprintParts: [
            CORE_FINDING_RULE_IDS.ENVIRONMENT_PENDING_MAPPING_SUGGESTION,
            suggestion.id,
          ],
        });
      }

      return { detections, evaluatedRuleIds };
    },
  };
}
