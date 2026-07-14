import { describe, expect, it } from "vitest";
import type { ConfigurationEntry } from "../src/domain/configuration.js";
import {
  isApprovedActionPlan,
  type ActionPlan,
  type ApprovedActionPlan,
} from "../src/domain/action.js";
import { actionPlanId, projectId } from "../src/index.js";

describe("@rayvan/core domain types", () => {
  it("compiles and constructs core domain objects", () => {
    const plan: ActionPlan = {
      id: actionPlanId("plan-1"),
      projectId: projectId("project-1"),
      pluginId: "vercel",
      status: "draft",
      summary: "Update environment variable",
      operations: [],
    };

    expect(plan.status).toBe("draft");
    expect(isApprovedActionPlan(plan)).toBe(false);
  });

  it("requires an approval record for approved execution", () => {
    const approved: ApprovedActionPlan = {
      id: actionPlanId("plan-2"),
      projectId: projectId("project-1"),
      pluginId: "github",
      status: "approved",
      summary: "Rotate deploy key",
      operations: [],
      approval: {
        id: "approval-1",
        actionPlanId: actionPlanId("plan-2"),
        approvedBy: "human",
        approvedAt: new Date().toISOString(),
      },
    };

    expect(isApprovedActionPlan(approved)).toBe(true);
  });

  it("does not expose plain secret values on configuration entries", () => {
    const entry: ConfigurationEntry = {
      id: "cfg-1" as ConfigurationEntry["id"],
      environmentId: "env-1" as ConfigurationEntry["environmentId"],
      integrationId: "int-1" as ConfigurationEntry["integrationId"],
      key: "DATABASE_URL",
      isSecret: true,
      valueFingerprint: "sha256:abc",
    };

    expect("value" in entry).toBe(false);
    expect(entry.valueFingerprint).toBeDefined();
  });
});
