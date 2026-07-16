import type { FindingDetection, FindingEvaluationScope } from "@rayvan/core";

import { CORE_FINDING_RULE_IDS } from "../rules/registry.js";
import type { FindingEvaluator, FindingEvaluatorResult } from "./types.js";

function inScope(
  scope: FindingEvaluationScope,
  item: {
    projectId?: string;
    environmentId?: string;
    connectionId?: string;
  },
): boolean {
  if (scope.type === "project") {
    return true;
  }
  if (scope.type === "environment") {
    return item.environmentId === scope.environmentId;
  }
  return item.connectionId === scope.connectionId;
}

/**
 * Change apply / verification / plan-stale findings.
 * When change plan / apply / verification data is unavailable (undefined),
 * those rules are omitted from evaluatedRuleIds so existing findings stay open.
 */
export function createChangeEvaluator(): FindingEvaluator {
  const ruleIds = [
    CORE_FINDING_RULE_IDS.CHANGE_APPLY_FAILED,
    CORE_FINDING_RULE_IDS.CHANGE_APPLY_PARTIAL,
    CORE_FINDING_RULE_IDS.CHANGE_VERIFICATION_FAILED,
    CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE,
  ] as const;

  return {
    id: "rayvan.change",
    ruleIds,
    evaluate({ projectId, scope, context }): FindingEvaluatorResult {
      const detections: FindingDetection[] = [];
      const evaluatedRuleIds: string[] = [];
      const applies = context.changeApplies;
      const verifications = context.changeVerifications;
      const plans = context.changePlans;

      if (applies !== undefined) {
        evaluatedRuleIds.push(
          CORE_FINDING_RULE_IDS.CHANGE_APPLY_FAILED,
          CORE_FINDING_RULE_IDS.CHANGE_APPLY_PARTIAL,
        );

        for (const apply of applies) {
          if (
            !inScope(scope, {
              projectId: apply.projectId,
              environmentId: apply.environmentId,
              connectionId: apply.connectionId,
            })
          ) {
            continue;
          }

          if (apply.status === "failed" || apply.status === "timed_out") {
            detections.push({
              ruleId: CORE_FINDING_RULE_IDS.CHANGE_APPLY_FAILED,
              projectId,
              severity: "error",
              title: "Change apply failed",
              summary:
                apply.safeErrorMessage ??
                `Change apply ${apply.id} failed for plan ${apply.changePlanId}.`,
              scope: {
                connectionId: apply.connectionId,
                resourceBindingId: apply.resourceBindingId,
                changePlanId: apply.changePlanId,
                environmentId: apply.environmentId,
              },
              evidence: [
                {
                  type: "message",
                  message:
                    apply.safeErrorMessage ??
                    `Apply ${apply.id} status=${apply.status}`,
                },
              ],
              fingerprintParts: [
                CORE_FINDING_RULE_IDS.CHANGE_APPLY_FAILED,
                apply.id,
              ],
              remediation: {
                type: "generate_change_plan",
                connectionId: apply.connectionId,
                resourceBindingId: apply.resourceBindingId,
                label: "Retry change plan",
              },
            });
          }

          if (apply.partial === true && apply.status === "succeeded") {
            detections.push({
              ruleId: CORE_FINDING_RULE_IDS.CHANGE_APPLY_PARTIAL,
              projectId,
              severity: "warning",
              title: "Change apply partial",
              summary: `Change apply ${apply.id} completed only partially.`,
              scope: {
                connectionId: apply.connectionId,
                resourceBindingId: apply.resourceBindingId,
                changePlanId: apply.changePlanId,
                environmentId: apply.environmentId,
              },
              evidence: [
                {
                  type: "message",
                  message: `Apply ${apply.id} reported partial success.`,
                },
              ],
              fingerprintParts: [
                CORE_FINDING_RULE_IDS.CHANGE_APPLY_PARTIAL,
                apply.id,
              ],
            });
          }
        }
      }

      if (verifications !== undefined) {
        evaluatedRuleIds.push(
          CORE_FINDING_RULE_IDS.CHANGE_VERIFICATION_FAILED,
        );
        const applyById = new Map(
          (applies ?? []).map((apply) => [apply.id, apply]),
        );
        for (const verification of verifications) {
          if (verification.status !== "failed") {
            continue;
          }
          const apply = applyById.get(verification.changeApplyId);
          if (
            apply &&
            !inScope(scope, {
              projectId: apply.projectId,
              environmentId: apply.environmentId,
              connectionId: apply.connectionId,
            })
          ) {
            continue;
          }
          if (!apply && scope.type !== "project") {
            continue;
          }
          detections.push({
            ruleId: CORE_FINDING_RULE_IDS.CHANGE_VERIFICATION_FAILED,
            projectId,
            severity: "error",
            title: "Change verification failed",
            summary:
              verification.safeErrorMessage ??
              `Verification failed for apply ${verification.changeApplyId}.`,
            scope: {
              connectionId: apply?.connectionId,
              resourceBindingId: apply?.resourceBindingId,
              changePlanId: apply?.changePlanId,
              environmentId: apply?.environmentId,
            },
            evidence: [
              {
                type: "message",
                message:
                  verification.safeErrorMessage ??
                  `Verification ${verification.id} failed.`,
              },
            ],
            fingerprintParts: [
              CORE_FINDING_RULE_IDS.CHANGE_VERIFICATION_FAILED,
              verification.id,
            ],
          });
        }
      }

      // Skip plan-stale when plan data is unavailable (undefined)
      if (plans === undefined) {
        return { detections, evaluatedRuleIds };
      }

      evaluatedRuleIds.push(CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE);

      for (const plan of plans) {
        if (
          !inScope(scope, {
            projectId: plan.projectId,
            environmentId: plan.environmentId,
            connectionId: plan.connectionId,
          })
        ) {
          continue;
        }
        if (
          plan.status === "applied" ||
          plan.status === "rejected" ||
          plan.status === "superseded" ||
          plan.status === "expired"
        ) {
          continue;
        }
        if (
          plan.observedStateChecksum &&
          plan.currentObservedChecksum &&
          plan.observedStateChecksum !== plan.currentObservedChecksum
        ) {
          detections.push({
            ruleId: CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE,
            projectId,
            severity: "warning",
            title: "Change plan stale",
            summary: `Change plan ${plan.id} is stale because observed state changed.`,
            scope: {
              connectionId: plan.connectionId,
              resourceBindingId: plan.resourceBindingId,
              changePlanId: plan.id,
              environmentId: plan.environmentId,
            },
            evidence: [
              {
                type: "message",
                message: `Plan checksum ${plan.observedStateChecksum} != current ${plan.currentObservedChecksum}.`,
              },
            ],
            fingerprintParts: [
              CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE,
              plan.id,
            ],
            remediation: {
              type: "generate_change_plan",
              connectionId: plan.connectionId,
              resourceBindingId: plan.resourceBindingId,
              label: "Regenerate change plan",
            },
          });
        }
      }

      return { detections, evaluatedRuleIds };
    },
  };
}
