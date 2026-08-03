import { describe, expect, it } from "vitest";
import { InProcessPluginRegistry, validatePlugin } from "@rayvan/plugin-sdk";

import {
  GITHUB_PLUGIN_ID,
  manifest,
  plugin,
  resetGithubFixtureStore,
} from "../src/index.js";

describe("@rayvan/plugin-github", () => {
  it("uses canonical plugin id and registers with full capabilities", () => {
    expect(manifest.id).toBe(GITHUB_PLUGIN_ID);
    expect(manifest.id).toBe("io.rayvan.github");
    expect(manifest.capabilities).toContain("discover");
    expect(manifest.setup?.authMethods).toEqual([
      "github_device_flow",
      "pat",
    ]);

    expect(() => validatePlugin(plugin)).not.toThrow();

    const registry = new InProcessPluginRegistry();
    registry.register(plugin);
    expect(registry.get(GITHUB_PLUGIN_ID)?.manifest.name).toBe("GitHub");
  });

  it("runs fixture discover → inspect → plan → apply → verify", async () => {
    resetGithubFixtureStore();
    process.env.RAYVAN_GITHUB_FIXTURE = "1";

    const discovered = await plugin.discover!({
      pluginId: GITHUB_PLUGIN_ID,
      integrationId: "conn-1",
      connectionMetadata: { fixture: true },
    });
    const variablesResource = discovered.find(
      (item) => item.resourceType === "github.actions_repository_variables",
    );
    expect(variablesResource).toBeTruthy();

    const binding = {
      resourceId: "binding-1",
      pluginId: GITHUB_PLUGIN_ID,
      providerResourceId: variablesResource!.providerResourceId,
      resourceType: variablesResource!.resourceType,
      projectId: "project-1",
    };

    const observed = await plugin.inspect!({
      pluginId: GITHUB_PLUGIN_ID,
      integrationId: "conn-1",
      resource: binding,
      connectionMetadata: { fixture: true },
    });
    expect(observed.attributes.variables).toEqual(
      expect.arrayContaining([
        { name: "NODE_VERSION", value: "20" },
      ]),
    );

    const plan = await plugin.plan!({
      pluginId: GITHUB_PLUGIN_ID,
      integrationId: "conn-1",
      resource: binding,
      observed,
      desired: {
        resourceId: binding.resourceId,
        pluginId: GITHUB_PLUGIN_ID,
        resourceType: binding.resourceType,
        attributes: {
          variables: {
            NODE_VERSION: "22",
            NEW_VAR: "hello",
          },
        },
      },
      connectionMetadata: { fixture: true },
    });
    expect(plan.operations.length).toBeGreaterThan(0);

    const approved = {
      plan,
      approvalId: "approval-1",
      approvedAt: new Date().toISOString(),
      approvedOperationIds: plan.operations.map((op) => op.id),
      destructiveApproval: false,
    };

    const applyResult = await plugin.apply!({
      pluginId: GITHUB_PLUGIN_ID,
      integrationId: "conn-1",
      resource: binding,
      approvedPlan: approved,
      connectionMetadata: { fixture: true },
    });
    expect(applyResult.ok).toBe(true);

    const verifyResult = await plugin.verify!({
      pluginId: GITHUB_PLUGIN_ID,
      integrationId: "conn-1",
      resource: binding,
      approvedPlan: approved,
      applyResult,
      connectionMetadata: { fixture: true },
    });
    expect(verifyResult.ok).toBe(true);

    delete process.env.RAYVAN_GITHUB_FIXTURE;
  });
});
