import { describe, expect, it } from "vitest";
import { actionPlanId, projectId, type ActionPlan } from "@rayvan/core";
import { assertExecutablePlan } from "../src/execution/index.js";

describe("@rayvan/action-engine", () => {
  it("cannot execute without an approval record", () => {
    const draftPlan: ActionPlan = {
      id: actionPlanId("plan-1"),
      projectId: projectId("project-1"),
      pluginId: "vercel",
      status: "approved",
      summary: "Deploy",
      operations: [],
    };

    expect(() => assertExecutablePlan(draftPlan as never)).toThrow(
      "explicit approval",
    );
  });
});
