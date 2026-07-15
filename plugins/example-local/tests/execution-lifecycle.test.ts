import { beforeEach, describe, expect, it } from "vitest";
import {
  createPluginExecutionStack,
  type ResourceBinding,
} from "@rayvan/plugin-sdk";
import {
  EXAMPLE_LOCAL_PLUGIN_ID,
  LOCAL_SERVICE_RESOURCE_TYPE,
  plugin,
  resetExampleLocalStore,
} from "../src/index.js";

const actor = {
  id: "tester",
  type: "user" as const,
};

const binding: ResourceBinding = {
  resourceId: "res-local-api",
  pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
  providerResourceId: "local-api",
  resourceType: LOCAL_SERVICE_RESOURCE_TYPE,
  projectId: "project-1",
  environmentId: "env-local",
};

describe("example-local via PluginExecutionService", () => {
  beforeEach(() => {
    resetExampleLocalStore();
  });

  it("runs discover → inspect → plan → approve → apply → verify through the stack", async () => {
    const { executionService } = createPluginExecutionStack({
      plugins: [plugin],
    });

    const discovered = await executionService.discover({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      actor,
      projectId: "project-1",
      environmentId: "env-local",
      context: {
        pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
        integrationId: "integration-1",
        projectId: "project-1",
        environmentId: "env-local",
      },
    });
    expect(discovered.status).toBe("succeeded");
    if (discovered.status !== "succeeded") {
      throw new Error(discovered.error?.message);
    }
    expect(discovered.data.map((resource) => resource.providerResourceId)).toEqual([
      "local-api",
      "local-worker",
    ]);

    const inspected = await executionService.inspect({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      actor,
      resourceId: binding.resourceId,
      context: {
        pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
        integrationId: "integration-1",
        resource: binding,
      },
    });
    expect(inspected.status).toBe("succeeded");
    if (inspected.status !== "succeeded") {
      throw new Error(inspected.error?.message);
    }
    expect(inspected.data.attributes.port).toBe(3000);

    const planned = await executionService.plan({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      actor,
      resourceId: binding.resourceId,
      context: {
        pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
        integrationId: "integration-1",
        resource: binding,
        observed: inspected.data,
        desired: {
          resourceId: binding.resourceId,
          pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
          resourceType: LOCAL_SERVICE_RESOURCE_TYPE,
          attributes: { port: 4000 },
        },
      },
    });
    expect(planned.status).toBe("succeeded");
    if (planned.status !== "succeeded") {
      throw new Error(planned.error?.message);
    }
    expect(planned.data.operations).toHaveLength(1);

    const approvedPlan = {
      plan: planned.data,
      approvalId: "approval-1",
      approvedAt: "1970-01-01T00:00:00.000Z",
      approvedOperationIds: planned.data.operations
        .filter((operation) => operation.requiresApproval)
        .map((operation) => operation.id),
      approvedBy: actor,
    };

    const applied = await executionService.apply({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      actor,
      resourceId: binding.resourceId,
      context: {
        pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
        integrationId: "integration-1",
        resource: binding,
        approvedPlan,
      },
    });
    expect(applied.status).toBe("succeeded");
    if (applied.status !== "succeeded") {
      throw new Error(applied.error?.message);
    }
    expect(applied.data.appliedOperationIds).toEqual(["set-port"]);
    expect(applied.data.resultingState?.attributes.port).toBe(4000);

    const verified = await executionService.verify({
      pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
      actor,
      resourceId: binding.resourceId,
      context: {
        pluginId: EXAMPLE_LOCAL_PLUGIN_ID,
        integrationId: "integration-1",
        resource: binding,
        approvedPlan,
        applyResult: applied.data,
      },
    });
    expect(verified.status).toBe("succeeded");
    if (verified.status !== "succeeded") {
      throw new Error(verified.error?.message);
    }
    expect(verified.data.ok).toBe(true);
    expect(verified.data.observed?.attributes.port).toBe(4000);
  });
});
