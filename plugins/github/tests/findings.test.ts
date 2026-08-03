import { describe, expect, it } from "vitest";

import { GITHUB_PLUGIN_ID, plugin } from "../src/index.js";

describe("github evaluate_findings", () => {
  it("emits unused and missing variable detections from observed payload", async () => {
    const result = await plugin.evaluateFindings!({
      pluginId: GITHUB_PLUGIN_ID,
      projectId: "project-1",
      connectionId: "conn-1",
      environments: [{ id: "env-1", name: "Production" }],
      resources: [
        {
          resourceBindingId: "binding-1",
          resourceType: "github.actions_repository_variables",
          providerResourceId: "acme/demo#actions-variables",
          environmentId: "env-1",
        },
      ],
      observedStates: [
        {
          resourceBindingId: "binding-1",
          value: {
            access: "readable",
            sensitive: false,
            value: JSON.stringify({
              variables: {
                NODE_VERSION: "20",
                UNUSED_FLAG: "1",
              },
              workflowVariableRefs: ["NODE_VERSION", "MISSING_VAR"],
            }),
          },
        },
      ],
    });

    const ruleIds = result.detections.map((item) => item.ruleId).sort();
    expect(ruleIds).toContain(`${GITHUB_PLUGIN_ID}.unused-actions-variable`);
    expect(ruleIds).toContain(`${GITHUB_PLUGIN_ID}.missing-referenced-variable`);
  });
});
