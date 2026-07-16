import type { FindingDetection, FindingEvaluationScope } from "@rayvan/core";

import { CORE_FINDING_RULE_IDS } from "../rules/registry.js";
import type {
  FindingsConnectionSnapshot,
  ProjectFindingsContext,
} from "../types.js";
import type { FindingEvaluator, FindingEvaluatorResult } from "./types.js";

function connectionsForScope(
  context: ProjectFindingsContext,
  scope: FindingEvaluationScope,
): FindingsConnectionSnapshot[] {
  if (scope.type === "connection") {
    return context.connections.filter(
      (connection) => connection.id === scope.connectionId,
    );
  }
  if (scope.type === "environment") {
    // Connections that have bindings in this environment
    const connectionIds = new Set(
      context.resourceBindings
        .filter((binding) => binding.environmentId === scope.environmentId)
        .map((binding) => binding.connectionId),
    );
    return context.connections.filter((connection) =>
      connectionIds.has(connection.id),
    );
  }
  return context.connections;
}

function isPermissionErrorCode(code: string | undefined): boolean {
  if (!code) {
    return false;
  }
  const lower = code.toLowerCase();
  return (
    lower.includes("permission") ||
    lower.includes("forbidden") ||
    lower.includes("unauthorized") ||
    lower === "insufficient_scope"
  );
}

export function createIntegrationEvaluator(): FindingEvaluator {
  const ruleIds = [
    CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
    CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_REVOKED,
    CORE_FINDING_RULE_IDS.INTEGRATION_AUTHENTICATION_FAILED,
    CORE_FINDING_RULE_IDS.INTEGRATION_PERMISSION_MISSING,
    CORE_FINDING_RULE_IDS.INTEGRATION_PLUGIN_DISABLED,
    CORE_FINDING_RULE_IDS.INTEGRATION_PLUGIN_INCOMPATIBLE,
    CORE_FINDING_RULE_IDS.INTEGRATION_SYNC_FAILURE,
    CORE_FINDING_RULE_IDS.INTEGRATION_CREDENTIAL_MISSING,
  ] as const;

  return {
    id: "rayvan.integration",
    ruleIds,
    evaluate({ projectId, scope, context }): FindingEvaluatorResult {
      const detections: FindingDetection[] = [];
      const connections = connectionsForScope(context, scope);
      const pluginById = new Map(
        context.installedPlugins.map((plugin) => [plugin.pluginId, plugin]),
      );

      for (const connection of connections) {
        const base = {
          connectionId: connection.id,
        };
        const evidence = [
          {
            type: "connection_error" as const,
            connectionId: connection.id,
            errorCode: connection.lastErrorCode,
            safeMessage: `${connection.name} is ${connection.status}`,
          },
        ];

        const push = (
          ruleId: string,
          title: string,
          summary: string,
          severity: FindingDetection["severity"],
          remediationType: "reauthenticate" | "open_integration" | "resync" = "open_integration",
        ) => {
          const remediation =
            remediationType === "reauthenticate"
              ? {
                  type: "reauthenticate" as const,
                  connectionId: connection.id,
                  label: `Reauthenticate ${connection.name}`,
                }
              : remediationType === "resync"
                ? {
                    type: "resync" as const,
                    connectionId: connection.id,
                    pluginId: connection.pluginId,
                    label: `Resync ${connection.name}`,
                  }
                : {
                    type: "open_integration" as const,
                    connectionId: connection.id,
                    label: `Open ${connection.name}`,
                  };
          detections.push({
            ruleId,
            projectId,
            severity,
            title,
            summary,
            scope: base,
            evidence,
            fingerprintParts: [ruleId, connection.id],
            remediation,
          });
        };

        if (connection.status === "expired") {
          push(
            CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
            "Connection expired",
            `${connection.name} connection has expired.`,
            "error",
            "reauthenticate",
          );
        }

        if (connection.status === "revoked") {
          push(
            CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_REVOKED,
            "Connection revoked",
            `${connection.name} connection has been revoked.`,
            "error",
            "reauthenticate",
          );
        }

        if (connection.status === "error") {
          if (isPermissionErrorCode(connection.lastErrorCode)) {
            push(
              CORE_FINDING_RULE_IDS.INTEGRATION_PERMISSION_MISSING,
              "Permission missing",
              `${connection.name} reported a permission error.`,
              "warning",
            );
          } else {
            push(
              CORE_FINDING_RULE_IDS.INTEGRATION_AUTHENTICATION_FAILED,
              "Authentication failed",
              `${connection.name} is in an error state.`,
              "error",
              "reauthenticate",
            );
          }
        }

        if (!connection.credentialReferenceId) {
          push(
            CORE_FINDING_RULE_IDS.INTEGRATION_CREDENTIAL_MISSING,
            "Credential missing",
            `${connection.name} has no credential reference.`,
            "error",
            "reauthenticate",
          );
        }

        const plugin = pluginById.get(connection.pluginId);
        if (plugin) {
          if (!plugin.enabled || plugin.status === "disabled") {
            push(
              CORE_FINDING_RULE_IDS.INTEGRATION_PLUGIN_DISABLED,
              "Plugin disabled",
              `Plugin ${connection.pluginId} is disabled for ${connection.name}.`,
              "warning",
            );
          }
          if (plugin.status === "incompatible") {
            push(
              CORE_FINDING_RULE_IDS.INTEGRATION_PLUGIN_INCOMPATIBLE,
              "Plugin incompatible",
              `Plugin ${connection.pluginId} is incompatible for ${connection.name}.`,
              "error",
            );
          }
        }

        if (connection.lastFailedSyncAt) {
          const failedMs = Date.parse(connection.lastFailedSyncAt);
          const successMs = connection.lastSuccessfulSyncAt
            ? Date.parse(connection.lastSuccessfulSyncAt)
            : Number.NEGATIVE_INFINITY;
          if (
            !Number.isNaN(failedMs) &&
            (Number.isNaN(successMs) || failedMs > successMs)
          ) {
            push(
              CORE_FINDING_RULE_IDS.INTEGRATION_SYNC_FAILURE,
              "Sync failure",
              `${connection.name} last sync failed without a more recent success.`,
              "warning",
              "resync",
            );
          }
        }
      }

      return { detections, evaluatedRuleIds: [...ruleIds] };
    },
  };
}
