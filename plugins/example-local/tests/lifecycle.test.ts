import { beforeEach, describe, expect, it } from "vitest";
import {
  InProcessPluginRegistry,
  validateApplyResult,
  validateChangePlan,
  validateVerificationResult,
  type ResourceBinding,
} from "@rayvan/plugin-sdk";
import {
  EXAMPLE_LOCAL_PLUGIN_ID,
  LOCAL_SERVICE_RESOURCE_TYPE,
  plugin,
  resetExampleLocalStore,
} from "../src/index.js";

const binding: ResourceBinding = {
  resourceId: "res-local-api",
  pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
  providerResourceId: "local-api",
  resourceType: LOCAL_SERVICE_RESOURCE_TYPE,
  projectId: "project-1",
  environmentId: "env-local",
};

describe("@rayvan/plugin-example-local", () => {
  beforeEach(() => {
    resetExampleLocalStore();
  });

  it("registers successfully and reports capabilities", () => {
    const registry = new InProcessPluginRegistry();
    registry.register(plugin);

    expect(registry.list()).toHaveLength(1);
    expect(registry.supports(EXAMPLE_LOCAL_PLUGIN_ID, "discover")).toBe(true);
    expect(registry.supports(EXAMPLE_LOCAL_PLUGIN_ID, "authenticate")).toBe(
      false,
    );
  });

  it("discovers deterministic local services", async () => {
    const discovered = await plugin.discover!({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      integrationId: "integration-1",
    });

    expect(discovered).toHaveLength(2);
    expect(discovered.map((resource) => resource.providerResourceId)).toEqual([
      "local-api",
      "local-worker",
    ]);
    expect(JSON.parse(JSON.stringify(discovered))).toEqual(discovered);
  });

  it("inspects a discovered service", async () => {
    const observed = await plugin.inspect!({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      integrationId: "integration-1",
      resource: binding,
    });

    expect(observed.status).toBe("ready");
    expect(observed.attributes.port).toBe(3000);
    expect(JSON.parse(JSON.stringify(observed))).toEqual(observed);
  });

  it("plans, applies, and verifies when Rayvan and provider ids differ", async () => {
    const observed = await plugin.inspect!({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      integrationId: "integration-1",
      resource: binding,
    });

    const changePlan = await plugin.plan!({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      integrationId: "integration-1",
      resource: binding,
      observed,
      desired: {
        resourceId: binding.resourceId,
        pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
        resourceType: LOCAL_SERVICE_RESOURCE_TYPE,
        attributes: { port: 4000 },
      },
    });

    expect(changePlan.resourceId).toBe("res-local-api");
    expect(changePlan.operations).toHaveLength(1);
    expect(changePlan.operations[0]?.after).toBe(4000);
    validateChangePlan(changePlan);
    expect(JSON.parse(JSON.stringify(changePlan))).toEqual(changePlan);

    const approvedPlan = {
      plan: changePlan,
      approvalId: "approval-1",
      approvedAt: "1970-01-01T00:00:00.000Z",
    };

    const applyResult = await plugin.apply!({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      integrationId: "integration-1",
      resource: binding,
      approvedPlan,
    });

    expect(applyResult.ok).toBe(true);
    expect(applyResult.appliedOperationIds).toEqual(["set-port"]);
    expect(applyResult.resultingState?.attributes.port).toBe(4000);
    expect(applyResult.resultingState?.attributes.providerResourceId).toBe(
      "local-api",
    );
    validateApplyResult(applyResult, EXAMPLE_LOCAL_PLUGIN_ID);
    expect(JSON.parse(JSON.stringify(applyResult))).toEqual(applyResult);

    const verification = await plugin.verify!({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      integrationId: "integration-1",
      resource: binding,
      approvedPlan,
      applyResult,
    });

    expect(verification.ok).toBe(true);
    expect(verification.observed?.attributes.port).toBe(4000);
    validateVerificationResult(verification, EXAMPLE_LOCAL_PLUGIN_ID);
    expect(JSON.parse(JSON.stringify(verification))).toEqual(verification);
  });
});
