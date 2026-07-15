import { describe, expect, it } from "vitest";

import { createDevEnvironmentsGateway } from "./dev-gateway.js";

describe("createDevEnvironmentsGateway", () => {
  it("seeds fixtures independently per project and does not cross-leak", async () => {
    const gateway = createDevEnvironmentsGateway();
    await gateway.ensureProjectSeeded("project-a");
    await gateway.ensureProjectSeeded("project-b");

    const envsA = await gateway.listEnvironments("project-a");
    const envsB = await gateway.listEnvironments("project-b");
    expect(envsA.length).toBeGreaterThan(0);
    expect(envsB.length).toBe(envsA.length);
    expect(envsA[0]?.id).not.toBe(envsB[0]?.id);

    const keysA = await gateway.listConfigurationKeys("project-a");
    const keysB = await gateway.listConfigurationKeys("project-b");
    expect(keysA.every((key) => key.projectId === "project-a")).toBe(true);
    expect(keysB.every((key) => key.projectId === "project-b")).toBe(true);
  });

  it("is idempotent for the same project id", async () => {
    const gateway = createDevEnvironmentsGateway();
    await gateway.ensureProjectSeeded("project-1");
    await gateway.ensureProjectSeeded("project-1");

    const environments = await gateway.listEnvironments("project-1");
    expect(environments.filter((env) => env.name === "Development")).toHaveLength(1);
  });

  it("awaits an in-flight seed so concurrent callers never refresh an empty store", async () => {
    const gateway = createDevEnvironmentsGateway();

    const first = gateway.ensureProjectSeeded("project-1");
    const second = gateway.ensureProjectSeeded("project-1");
    await Promise.all([first, second]);

    const environments = await gateway.listEnvironments("project-1");
    expect(environments.map((env) => env.name)).toEqual(
      expect.arrayContaining([
        "Development",
        "Staging",
        "Production",
        "Preview",
        "Local Scratch",
      ]),
    );
    expect(environments.filter((env) => env.name === "Production")).toHaveLength(1);
  });

  it("creates a local_only environment and rejects duplicate names", async () => {
    const gateway = createDevEnvironmentsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const created = await gateway.createEnvironment({
      projectId: "project-1",
      name: "QA Lab",
      kind: "test",
    });
    expect(created.status).toBe("local_only");

    await expect(
      gateway.createEnvironment({
        projectId: "project-1",
        name: "QA Lab",
        kind: "test",
      }),
    ).rejects.toThrow(/already exists/i);
  });

  it("does not auto-accept mapping suggestions on seed", async () => {
    const gateway = createDevEnvironmentsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const pending = await gateway.listPendingSuggestions("project-1");
    expect(pending.length).toBeGreaterThan(0);
    expect(pending.every((item) => item.status === "pending")).toBe(true);

    const bindings = await gateway.listBindings("project-1");
    for (const suggestion of pending) {
      const bound = bindings.some(
        (binding) =>
          binding.discoveredResourceId === suggestion.discoveredResourceId &&
          binding.bindingStatus === "active",
      );
      expect(bound).toBe(false);
    }
  });

  it("rejects accepting a suggestion onto an environment from another project", async () => {
    const gateway = createDevEnvironmentsGateway();
    await gateway.ensureProjectSeeded("project-1");
    await gateway.ensureProjectSeeded("project-2");

    const pending = await gateway.listPendingSuggestions("project-1");
    const suggestion = pending[0];
    expect(suggestion).toBeDefined();
    const foreignEnv = (await gateway.listEnvironments("project-2"))[0];
    expect(foreignEnv).toBeDefined();

    await expect(
      gateway.acceptSuggestion({
        suggestionId: suggestion!.id,
        environmentId: foreignEnv!.id,
      }),
    ).rejects.toThrow(/does not belong to suggestion project/i);
  });

  it("accept binds a resource and reject does not", async () => {
    const gateway = createDevEnvironmentsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const pending = await gateway.listPendingSuggestions("project-1");
    const first = pending[0];
    const second = pending[1];
    expect(first).toBeDefined();
    expect(second).toBeDefined();

    const accepted = await gateway.acceptSuggestion({
      suggestionId: first!.id,
      environmentId: first!.suggestedEnvironmentId,
    });
    expect(accepted.suggestion.status).toBe("accepted");
    expect(accepted.binding?.bindingStatus).toBe("active");
    expect(accepted.binding?.discoveredResourceId).toBe(first!.discoveredResourceId);

    const rejected = await gateway.rejectSuggestion(second!.id);
    expect(rejected.status).toBe("rejected");

    const bindings = await gateway.listBindings("project-1");
    expect(
      bindings.some(
        (binding) =>
          binding.discoveredResourceId === second!.discoveredResourceId &&
          binding.bindingStatus === "active",
      ),
    ).toBe(false);
  });

  it("attaches, moves, and detaches resources without deleting discovered records", async () => {
    const gateway = createDevEnvironmentsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const environments = await gateway.listEnvironments("project-1");
    const development = environments.find((env) => env.name === "Development");
    const staging = environments.find((env) => env.name === "Staging");
    expect(development).toBeDefined();
    expect(staging).toBeDefined();

    const resources = await gateway.listDiscoveredResources("project-1");
    const unmapped = resources.find((resource) => resource.name === "GitHub develop");
    expect(unmapped).toBeDefined();

    const attached = await gateway.attachResource({
      projectId: "project-1",
      environmentId: development!.id,
      discoveredResourceId: unmapped!.id,
    });
    expect(attached.environmentId).toBe(development!.id);
    expect(attached.bindingStatus).toBe("active");

    const moved = await gateway.moveResource({
      bindingId: attached.id,
      environmentId: staging!.id,
    });
    expect(moved.environmentId).toBe(staging!.id);

    const detached = await gateway.detachResource(moved.id);
    expect(detached.bindingStatus).toBe("detached");

    const stillThere = (await gateway.listDiscoveredResources("project-1")).find(
      (resource) => resource.id === unmapped!.id,
    );
    expect(stillThere).toBeDefined();
  });

  it("never stores plaintext secrets in ordinary occurrence observedValue", async () => {
    const gateway = createDevEnvironmentsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const keys = await gateway.listConfigurationKeys("project-1");
    const sensitiveIds = new Set(
      keys.filter((key) => key.sensitive).map((key) => key.id),
    );
    const occurrences = await gateway.listOccurrences("project-1");

    for (const occurrence of occurrences) {
      if (!sensitiveIds.has(occurrence.configurationKeyId)) {
        continue;
      }
      expect(occurrence.observedValue).toBeUndefined();
      expect(
        occurrence.secretValueRef ||
          occurrence.maskedValue ||
          occurrence.valueAccess === "name_only" ||
          occurrence.valueAccess === "locked" ||
          occurrence.valueAccess === "missing",
      ).toBeTruthy();
    }
  });

  it("runs a read-only sync that does not auto-accept suggestions", async () => {
    const gateway = createDevEnvironmentsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const beforePending = await gateway.listPendingSuggestions("project-1");
    const result = await gateway.syncWithIntegrations("project-1");

    expect(result.cancelled).toBe(false);
    expect(result.plugins.length).toBeGreaterThan(0);

    const afterPending = await gateway.listPendingSuggestions("project-1");
    expect(afterPending.length).toBeGreaterThanOrEqual(beforePending.length);
    expect(afterPending.every((item) => item.status === "pending")).toBe(true);
  });

  it("preserves successful plugin results when one plugin fails", async () => {
    const gateway = createDevEnvironmentsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const result = await gateway.syncWithIntegrations("project-1");
    const failed = result.plugins.filter((plugin) => plugin.status === "failed");
    const succeeded = result.plugins.filter((plugin) => plugin.status === "success");

    expect(failed.length).toBeGreaterThanOrEqual(1);
    expect(succeeded.length).toBeGreaterThan(0);
    expect(result.resourcesTouched).toBeGreaterThan(0);
  });

  it("builds a configuration matrix with expected cell statuses", async () => {
    const gateway = createDevEnvironmentsGateway();
    await gateway.ensureProjectSeeded("project-1");

    const matrix = await gateway.getMatrix("project-1");
    expect(matrix.columns.length).toBeGreaterThan(0);
    expect(matrix.rows.length).toBeGreaterThan(0);
    expect(matrix.summary.missingCellCount).toBeGreaterThan(0);

    const stripeRow = matrix.rows.find((row) => row.name === "STRIPE_SECRET_KEY");
    expect(stripeRow).toBeDefined();
    expect(stripeRow!.cells.some((cell) => cell.status === "locked")).toBe(true);
    expect(stripeRow!.cells.some((cell) => cell.requiredMissing || cell.status === "missing")).toBe(
      true,
    );
  });

  it("creates a fresh isolated instance per factory call", async () => {
    const gatewayA = createDevEnvironmentsGateway();
    const gatewayB = createDevEnvironmentsGateway();
    await gatewayA.ensureProjectSeeded("project-1");

    const envsA = await gatewayA.listEnvironments("project-1");
    const envsB = await gatewayB.listEnvironments("project-1");
    expect(envsA.length).toBeGreaterThan(0);
    expect(envsB).toHaveLength(0);
  });
});
