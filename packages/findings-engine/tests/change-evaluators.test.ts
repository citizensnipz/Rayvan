import { describe, expect, it } from "vitest";

import { createChangeEvaluator } from "../src/evaluators/change.js";
import { CORE_FINDING_RULE_IDS } from "../src/rules/registry.js";
import { NOW, PROJECT, emptyContext } from "./helpers.js";

const evaluator = createChangeEvaluator();

describe("change evaluators", () => {
  it("omits plan-stale from evaluatedRuleIds when changePlans is undefined", async () => {
    const result = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        changeApplies: [],
        changeVerifications: [],
        // changePlans omitted
      }),
      now: NOW,
    });

    expect(result.evaluatedRuleIds).toContain(
      CORE_FINDING_RULE_IDS.CHANGE_APPLY_FAILED,
    );
    expect(result.evaluatedRuleIds).not.toContain(
      CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE,
    );
    expect(
      result.detections.some(
        (d) => d.ruleId === CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE,
      ),
    ).toBe(false);
  });

  it("includes plan-stale in evaluatedRuleIds when changePlans is an empty array", async () => {
    const result = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        changeApplies: [],
        changeVerifications: [],
        changePlans: [],
      }),
      now: NOW,
    });

    expect(result.evaluatedRuleIds).toContain(
      CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE,
    );
  });

  it("emits plan-stale when checksums disagree", async () => {
    const result = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        changePlans: [
          {
            id: "plan-1",
            pluginId: "vercel",
            connectionId: "conn-1",
            projectId: PROJECT,
            resourceBindingId: "binding-1",
            status: "approved",
            createdAt: NOW,
            observedStateChecksum: "abc",
            currentObservedChecksum: "def",
          },
        ],
      }),
      now: NOW,
    });

    expect(
      result.detections.some(
        (d) => d.ruleId === CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE,
      ),
    ).toBe(true);
    expect(result.evaluatedRuleIds).toContain(
      CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE,
    );
  });
});
