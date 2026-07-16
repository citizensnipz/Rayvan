import {
  DEFAULT_STALE_AFTER_MS,
  deriveEnvironmentStatus,
} from "@rayvan/config-engine";
import type {
  FindingDetection,
  FindingEvaluationScope,
} from "@rayvan/core";

import { CORE_FINDING_RULE_IDS } from "../rules/registry.js";
import type { ProjectFindingsContext } from "../types.js";
import type { FindingEvaluator, FindingEvaluatorResult } from "./types.js";

function environmentsForScope(
  context: ProjectFindingsContext,
  scope: FindingEvaluationScope,
) {
  if (scope.type === "environment") {
    return context.environments.filter(
      (environment) => environment.id === scope.environmentId,
    );
  }
  return context.environments;
}

function scopeMatchesEnvironment(
  scope: FindingEvaluationScope,
  environmentId: string,
): boolean {
  if (scope.type === "environment") {
    return scope.environmentId === environmentId;
  }
  if (scope.type === "connection") {
    return false;
  }
  return true;
}

/**
 * Configuration findings from persisted desired/observed/applied state.
 * Uses config-engine status helpers. Does NOT emit findings for editor drafts.
 * Stale inspection threshold matches config-engine DEFAULT_STALE_AFTER_MS (7 days).
 */
export function createConfigurationEvaluator(): FindingEvaluator {
  const ruleIds = [
    CORE_FINDING_RULE_IDS.CONFIGURATION_MISSING_REQUIRED,
    CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
    CORE_FINDING_RULE_IDS.CONFIGURATION_UNAPPLIED,
    CORE_FINDING_RULE_IDS.CONFIGURATION_REMOTE_CHANGED,
    CORE_FINDING_RULE_IDS.CONFIGURATION_PARTIALLY_APPLIED,
    CORE_FINDING_RULE_IDS.CONFIGURATION_COMPARISON_UNAVAILABLE,
    CORE_FINDING_RULE_IDS.CONFIGURATION_UNMANAGED,
    CORE_FINDING_RULE_IDS.CONFIGURATION_INSPECTION_STALE,
  ] as const;

  return {
    id: "rayvan.configuration",
    ruleIds,
    evaluate({ projectId, scope, context, now }): FindingEvaluatorResult {
      // Connection-scoped runs do not evaluate configuration rules.
      if (scope.type === "connection") {
        return { detections: [], evaluatedRuleIds: [] };
      }

      const detections: FindingDetection[] = [];
      const environments = environmentsForScope(context, scope);

      for (const environment of environments) {
        if (!scopeMatchesEnvironment(scope, environment.id)) {
          continue;
        }

        const envStatus = deriveEnvironmentStatus({
          environmentId: environment.id,
          keys: context.keys,
          desired: context.desired.filter(
            (value) => value.environmentId === environment.id,
          ),
          occurrences: context.occurrences.filter(
            (occurrence) => occurrence.environmentId === environment.id,
          ),
          applied: context.applied.filter(
            (state) => state.environmentId === environment.id,
          ),
          // Intentionally omit drafts — unsaved editor state must not create findings.
          now,
          staleAfterMs: DEFAULT_STALE_AFTER_MS,
        });

        for (const keyStatus of envStatus.keyStatuses) {
          // Drafts are intentionally omitted from deriveEnvironmentStatus input,
          // so editorDirty never drives findings — only persisted syncStatus does.
          const keyName = keyStatus.configurationKeyName;
          const key = context.keys.find(
            (item) => item.id === keyStatus.configurationKeyId,
          );
          const baseScope = {
            environmentId: environment.id,
            configurationKeyId: keyStatus.configurationKeyId,
          };
          const evidence = [
            {
              type: "configuration_comparison" as const,
              configurationKeyId: String(keyStatus.configurationKeyId),
              environmentId: environment.id,
              observedStates: [],
            },
          ];

          const push = (
            ruleId: string,
            title: string,
            summary: string,
            severity?: FindingDetection["severity"],
          ) => {
            detections.push({
              ruleId,
              projectId,
              severity,
              title,
              summary,
              scope: baseScope,
              evidence,
              fingerprintParts: [
                ruleId,
                environment.id,
                String(keyStatus.configurationKeyId),
              ],
              remediation: {
                type: "open_environment",
                environmentId: environment.id,
                label: `Open ${environment.name}`,
              },
            });
          };

          switch (keyStatus.syncStatus) {
            case "missing_remote":
              if (key?.required) {
                push(
                  CORE_FINDING_RULE_IDS.CONFIGURATION_MISSING_REQUIRED,
                  "Missing required configuration",
                  `${keyName} is required but missing in ${environment.name}.`,
                  "error",
                );
              }
              break;
            case "mismatched":
              push(
                CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
                "Configuration mismatch",
                `${keyName} desired value disagrees with observed values in ${environment.name}.`,
                "warning",
              );
              break;
            case "local_changes":
              push(
                CORE_FINDING_RULE_IDS.CONFIGURATION_UNAPPLIED,
                "Unapplied configuration changes",
                `${keyName} has saved desired changes not yet applied in ${environment.name}.`,
                "warning",
              );
              break;
            case "remote_changed":
              push(
                CORE_FINDING_RULE_IDS.CONFIGURATION_REMOTE_CHANGED,
                "Remote configuration changed",
                `${keyName} changed remotely in ${environment.name} while desired still matches last-applied.`,
                "warning",
              );
              break;
            case "partially_applied":
              push(
                CORE_FINDING_RULE_IDS.CONFIGURATION_PARTIALLY_APPLIED,
                "Partially applied configuration",
                `${keyName} is only partially applied across resources in ${environment.name}.`,
                "warning",
              );
              break;
            case "locked":
              push(
                CORE_FINDING_RULE_IDS.CONFIGURATION_COMPARISON_UNAVAILABLE,
                "Configuration comparison unavailable",
                `${keyName} values are locked or inaccessible in ${environment.name}.`,
                "info",
              );
              break;
            case "not_managed":
            case "missing_local":
              push(
                CORE_FINDING_RULE_IDS.CONFIGURATION_UNMANAGED,
                "Unmanaged configuration",
                `${keyName} is observed in ${environment.name} without a saved desired value.`,
                "info",
              );
              break;
            default:
              break;
          }

          if (keyStatus.observedMayBeStale) {
            push(
              CORE_FINDING_RULE_IDS.CONFIGURATION_INSPECTION_STALE,
              "Stale configuration inspection",
              `${keyName} observation in ${environment.name} may be stale (default threshold: 7 days).`,
              "info",
            );
          }
        }
      }

      return { detections, evaluatedRuleIds: [...ruleIds] };
    },
  };
}
