import type { FindingDetection } from "@rayvan/core";
import { findingEvaluationRunId } from "@rayvan/core";
import { describe, expect, it } from "vitest";

import { matchAndApplyDetections } from "../src/engine/dedupe.js";
import { buildFindingFingerprint } from "../src/fingerprint.js";
import {
  CORE_FINDING_RULE_IDS,
  getCoreFindingRule,
  listCoreFindingRules,
} from "../src/rules/registry.js";
import { makeFindingRecord, NOW, PROJECT } from "./helpers.js";

const ACTOR = { kind: "system" as const, id: "test" };

function detection(
  overrides: Partial<FindingDetection> = {},
): FindingDetection {
  return {
    ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
    projectId: PROJECT,
    title: "Configuration mismatch",
    summary: "Values disagree",
    scope: { environmentId: "env-dev", configurationKeyId: "key-1" },
    evidence: [{ type: "message", message: "mismatch" }],
    fingerprintParts: [
      CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      "env-dev",
      "key-1",
    ],
    ...overrides,
  };
}

describe("matchAndApplyDetections", () => {
  const rulesById = new Map(
    listCoreFindingRules().map((rule) => [rule.id, rule]),
  );

  it("does not duplicate when wording changes", () => {
    const det = detection();
    const fingerprint = buildFindingFingerprint({
      ruleId: det.ruleId,
      projectId: String(det.projectId),
      fingerprintParts: det.fingerprintParts,
    });
    const existing = makeFindingRecord({
      ruleId: det.ruleId,
      fingerprint,
      title: "Old title wording",
      summary: "Old summary",
      occurrenceCount: 2,
    });

    const result = matchAndApplyDetections({
      detections: [
        detection({
          title: "Brand new title wording",
          summary: "Completely different summary text",
        }),
      ],
      existing: [existing],
      rulesById,
      severityOverrides: new Map(),
      actor: ACTOR,
      now: NOW,
      evaluationRunId: findingEvaluationRunId("run-1"),
      succeededRuleIds: new Set([det.ruleId]),
      existingInScope: [existing],
    });

    expect(result.created).toHaveLength(0);
    expect(result.updated).toHaveLength(1);
    expect(result.updated[0]!.title).toBe("Brand new title wording");
    expect(result.updated[0]!.occurrenceCount).toBe(3);
    expect(result.updated[0]!.fingerprint).toBe(fingerprint);
  });

  it("reopens after resolve when fingerprint returns", () => {
    const det = detection();
    const fingerprint = buildFindingFingerprint({
      ruleId: det.ruleId,
      projectId: String(det.projectId),
      fingerprintParts: det.fingerprintParts,
    });
    const existing = makeFindingRecord({
      ruleId: det.ruleId,
      fingerprint,
      status: "resolved",
      resolvedAt: "2026-07-01T00:00:00.000Z",
      resolution: {
        source: "automatic",
        resolvedBy: ACTOR,
      },
    });

    const result = matchAndApplyDetections({
      detections: [det],
      existing: [existing],
      rulesById,
      severityOverrides: new Map(),
      actor: ACTOR,
      now: NOW,
      evaluationRunId: findingEvaluationRunId("run-2"),
      succeededRuleIds: new Set([det.ruleId]),
      existingInScope: [],
    });

    expect(result.reopened).toHaveLength(1);
    expect(result.reopened[0]!.status).toBe("open");
    expect(result.reopened[0]!.resolvedAt).toBeUndefined();
  });

  it("creates a new finding when fingerprint is new", () => {
    const det = detection();
    const result = matchAndApplyDetections({
      detections: [det],
      existing: [],
      rulesById,
      severityOverrides: new Map(),
      actor: ACTOR,
      now: NOW,
      evaluationRunId: findingEvaluationRunId("run-3"),
      succeededRuleIds: new Set([det.ruleId]),
      existingInScope: [],
    });
    expect(result.created).toHaveLength(1);
    expect(getCoreFindingRule(result.created[0]!.ruleId)?.category).toBe(
      "drift",
    );
  });
});
