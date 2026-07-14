import type { ChangePlan, PluginManifest } from "@rayvan/plugin-sdk";
import { describe, expect, it } from "vitest";

import {
  ChangeApprovalService,
  ChangePlanService,
  createInMemoryPluginPersistence,
  createPersistentExecutionEventSink,
  createPersistentPermissionResolver,
  DevelopmentMemoryCredentialStore,
  EnvironmentMappingService,
  OptimisticConcurrencyError,
  PersistentPluginPermissionResolver,
  PluginConnectionService,
  PluginDomainError,
  PluginExecutionGuard,
  PluginInstallationService,
  PluginPermissionService,
  ResourceBindingService,
  ResourceDiscoveryService,
  ResourceStateService,
} from "../src/plugins/index.js";

const actor = { type: "user" as const, id: "user-1", displayName: "Ada" };

function manifest(
  overrides: Partial<PluginManifest> = {},
): PluginManifest {
  return {
    id: "example.local",
    name: "Example Local",
    version: "1.0.0",
    publisher: "rayvan",
    rayvanApiVersion: "1",
    capabilities: ["discover", "inspect", "plan", "apply", "verify"],
    permissions: ["network"],
    resourceTypes: [
      {
        id: "local.service",
        name: "Local Service",
        schemaVersion: "1",
      },
    ],
    ...overrides,
  };
}

function createStack() {
  const db = createInMemoryPluginPersistence();
  const credentials = new DevelopmentMemoryCredentialStore();
  const installation = new PluginInstallationService(db.installedPlugins);
  const connections = new PluginConnectionService(
    db.installedPlugins,
    db.connections,
    db.credentialReferences,
    db.permissionGrants,
    db.resourceBindings,
    credentials,
  );
  const permissions = new PluginPermissionService(
    db.connections,
    db.permissionGrants,
  );
  const discovery = new ResourceDiscoveryService(
    db.connections,
    db.discoveredResources,
  );
  const bindings = new ResourceBindingService(
    db.discoveredResources,
    db.resourceBindings,
  );
  const mapping = new EnvironmentMappingService(db.mappingSuggestions);
  const state = new ResourceStateService(db.observedState, db.desiredState);
  const plans = new ChangePlanService(db.changePlans);
  const approvals = new ChangeApprovalService(
    db.changePlans,
    db.changePlanApprovals,
    db.changeApplies,
    db.changeVerifications,
  );
  const guard = new PluginExecutionGuard(db.installedPlugins, db.connections);
  return {
    db,
    credentials,
    installation,
    connections,
    permissions,
    discovery,
    bindings,
    mapping,
    state,
    plans,
    approvals,
    guard,
  };
}

describe("Installed plugins", () => {
  it("registers built-ins, updates versions, and marks missing/incompatible", async () => {
    const { installation, db } = createStack();

    const first = await installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    expect(first).toHaveLength(1);
    expect(first[0]?.enabled).toBe(true);
    expect(first[0]?.status).toBe("installed");

    const updated = await installation.reconcileBuiltIns([
      { manifest: manifest({ version: "1.1.0" }) },
    ]);
    expect(updated[0]?.pluginVersion).toBe("1.1.0");
    expect(updated[0]?.manifestSnapshot.version).toBe("1.1.0");

    const incompatible = await installation.reconcileBuiltIns([
      { manifest: manifest({ version: "1.2.0", rayvanApiVersion: "999" }) },
    ]);
    expect(incompatible[0]?.status).toBe("incompatible");
    expect(incompatible[0]?.enabled).toBe(false);

    const recovered = await installation.reconcileBuiltIns([
      { manifest: manifest({ version: "1.3.0" }) },
    ]);
    expect(recovered[0]?.status).toBe("installed");
    expect(recovered[0]?.enabled).toBe(false);

    await installation.reconcileBuiltIns([]);
    const missing = await db.installedPlugins.getByPluginId("example.local");
    expect(missing?.status).toBe("missing");
    expect(missing?.enabled).toBe(false);
  });

  it("disabled plugin prevents execution via guard", async () => {
    const { installation, guard } = createStack();
    const [installed] = await installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    await installation.disable(installed!.id);

    const result = await guard.assertExecutable({ pluginId: "example.local" });
    expect(result.ok).toBe(false);
    expect(result.reason).toContain("plugin_not_executable");
  });

  it("soft uninstall preserves records and blocks execution", async () => {
    const { installation, guard, db } = createStack();
    const [installed] = await installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    const uninstalled = await installation.uninstall(installed!.id);
    expect(uninstalled.status).toBe("disabled");
    expect(uninstalled.enabled).toBe(false);
    expect(await db.installedPlugins.getById(installed!.id)).toBeTruthy();
    expect(
      (await guard.assertExecutable({ pluginId: "example.local" })).ok,
    ).toBe(false);
  });
});

describe("Connections", () => {
  it("creates multiple connections and associates credentials", async () => {
    const { installation, connections, credentials, db } = createStack();
    const [installed] = await installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);

    const personal = await connections.create({
      installedPluginId: installed!.id,
      name: "Personal",
      projectId: "project-a",
    });
    const company = await connections.create({
      installedPluginId: installed!.id,
      name: "Company",
      projectId: "project-b",
    });

    const reference = await credentials.put({
      pluginId: "example.local",
      connectionId: personal.id,
      provider: "development_memory",
      credentialType: "token",
      secret: { token: "secret-value" },
    });
    const withCred = await connections.attachCredentialReference(
      personal.id,
      reference,
    );

    expect(withCred.credentialReferenceId).toBe(reference.id);
    expect(await credentials.get(reference)).toEqual({ token: "secret-value" });
    expect(await connections.listByPluginId("example.local")).toHaveLength(2);
    expect(await connections.listByProjectId("project-a")).toEqual([
      expect.objectContaining({ id: personal.id }),
    ]);
    expect(company.id).not.toBe(personal.id);

    const refs = await db.credentialReferences.listByConnectionId(personal.id);
    expect(refs[0]?.storageKey).toBeTruthy();
    expect(JSON.stringify(refs)).not.toContain("secret-value");
  });

  it("disconnect revokes grants, invalidates bindings, deletes credentials, preserves history", async () => {
    const stack = createStack();
    const [installed] = await stack.installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    const connection = await stack.connections.create({
      installedPluginId: installed!.id,
      name: "Conn",
      projectId: "project-a",
      status: "connected",
    });
    const reference = await stack.credentials.put({
      pluginId: "example.local",
      connectionId: connection.id,
      provider: "development_memory",
      credentialType: "token",
      secret: { token: "abc" },
    });
    await stack.connections.attachCredentialReference(connection.id, reference);
    await stack.permissions.grant({
      pluginId: "example.local",
      connectionId: connection.id,
      permissions: ["network"],
      grantedBy: actor,
    });
    const discovered = await stack.discovery.sync({
      connectionId: connection.id,
      installedPluginId: installed!.id,
      pluginId: "example.local",
      items: [
        {
          providerResourceId: "svc-1",
          resourceType: "local.service",
          name: "API",
          metadata: {},
          pluginVersion: "1.0.0",
          schemaVersion: "1",
        },
      ],
    });
    const binding = await stack.bindings.bind({
      projectId: "project-a",
      discoveredResourceId: discovered[0]!.id,
      createdBy: actor,
    });
    await stack.db.executionHistory.append({
      id: "hist-1",
      executionId: "exec-1",
      pluginId: "example.local",
      pluginVersion: "1.0.0",
      capability: "discover",
      status: "succeeded",
      startedAt: "2026-01-01T00:00:00.000Z",
      finishedAt: "2026-01-01T00:00:01.000Z",
      durationMs: 1000,
      actor,
      connectionId: connection.id,
      warningCount: 0,
      recordedAt: "2026-01-01T00:00:01.000Z",
    });

    const disconnected = await stack.connections.disconnect(connection.id);
    expect(disconnected.status).toBe("disconnected");
    expect(await stack.credentials.exists(reference)).toBe(false);
    expect(
      await stack.db.permissionGrants.listActiveByConnectionId(connection.id),
    ).toHaveLength(0);
    expect(
      (await stack.db.resourceBindings.getById(binding.id))?.bindingStatus,
    ).toBe("invalid");
    expect(
      await stack.db.discoveredResources.listByConnectionId(connection.id),
    ).toHaveLength(1);
    expect(
      await stack.db.executionHistory.listByConnectionId(connection.id),
    ).toHaveLength(1);
  });
});

describe("Permissions", () => {
  it("persists, scopes, revokes, and ignores disconnected connections", async () => {
    const stack = createStack();
    const [installed] = await stack.installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    const connection = await stack.connections.create({
      installedPluginId: installed!.id,
      name: "Conn",
      status: "connected",
    });

    await stack.permissions.grant({
      pluginId: "example.local",
      connectionId: connection.id,
      permissions: ["network", "read_secrets"],
      projectId: "project-a",
      environmentId: "env-prod",
      grantedBy: actor,
    });

    const active = await stack.permissions.listActive(connection.id);
    expect(active.map((grant) => grant.permission).sort()).toEqual([
      "network",
      "read_secrets",
    ]);

    const resolver = new PersistentPluginPermissionResolver(
      stack.db.connections,
      stack.db.permissionGrants,
      { connectionId: connection.id },
    );
    const envScoped = await resolver.resolve({
      pluginId: "example.local",
      capability: "discover",
      actor,
      projectId: "project-a",
      environmentId: "env-prod",
    });
    expect(envScoped).toEqual(
      expect.arrayContaining(["network", "read_secrets"]),
    );

    const projectOnly = await resolver.resolve({
      pluginId: "example.local",
      capability: "discover",
      actor,
      projectId: "project-a",
    });
    expect(projectOnly).toEqual([]);

    await stack.permissions.revoke({
      grantId: active[0]!.id,
      revokedBy: actor,
      reason: "no longer needed",
    });

    await stack.connections.disconnect(connection.id);
    const afterDisconnect = await resolver.resolve({
      pluginId: "example.local",
      capability: "discover",
      actor,
      projectId: "project-a",
      environmentId: "env-prod",
    });
    expect(afterDisconnect).toEqual([]);
  });

  it("replaces grants only within the same project/environment scope", async () => {
    const stack = createStack();
    const [installed] = await stack.installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    const connection = await stack.connections.create({
      installedPluginId: installed!.id,
      name: "Conn",
      status: "connected",
    });

    await stack.permissions.grant({
      pluginId: "example.local",
      connectionId: connection.id,
      permissions: ["network"],
      projectId: "project-a",
      grantedBy: actor,
    });
    await stack.permissions.grant({
      pluginId: "example.local",
      connectionId: connection.id,
      permissions: ["read_secrets"],
      projectId: "project-b",
      environmentId: "env-prod",
      grantedBy: actor,
    });

    await stack.permissions.grant({
      pluginId: "example.local",
      connectionId: connection.id,
      permissions: ["write_remote_configuration"],
      projectId: "project-a",
      grantedBy: actor,
    });

    const active = await stack.permissions.listActive(connection.id);
    expect(active).toHaveLength(2);
    expect(
      active.find((grant) => grant.projectId === "project-a")?.permission,
    ).toBe("write_remote_configuration");
    expect(
      active.find((grant) => grant.projectId === "project-b")?.permission,
    ).toBe("read_secrets");
  });
});

describe("Discovery", () => {
  it("upserts, marks missing, rediscovers, and keeps connections distinct", async () => {
    const stack = createStack();
    const [installed] = await stack.installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    const a = await stack.connections.create({
      installedPluginId: installed!.id,
      name: "A",
      status: "connected",
    });
    const b = await stack.connections.create({
      installedPluginId: installed!.id,
      name: "B",
      status: "connected",
    });

    const first = await stack.discovery.sync({
      connectionId: a.id,
      installedPluginId: installed!.id,
      pluginId: "example.local",
      items: [
        {
          providerResourceId: "svc-1",
          resourceType: "local.service",
          name: "One",
          metadata: { v: 1 },
          pluginVersion: "1.0.0",
          schemaVersion: "1",
        },
        {
          providerResourceId: "svc-2",
          resourceType: "local.service",
          name: "Two",
          metadata: {},
          pluginVersion: "1.0.0",
          schemaVersion: "1",
        },
      ],
    });
    expect(first).toHaveLength(2);

    const second = await stack.discovery.sync({
      connectionId: a.id,
      installedPluginId: installed!.id,
      pluginId: "example.local",
      items: [
        {
          providerResourceId: "svc-1",
          resourceType: "local.service",
          name: "One-updated",
          metadata: { v: 2 },
          pluginVersion: "1.0.0",
          schemaVersion: "1",
        },
      ],
    });
    expect(second[0]?.name).toBe("One-updated");
    expect(second[0]?.id).toBe(first[0]?.id);

    const listed = await stack.discovery.listByConnectionId(a.id);
    const missing = listed.find((item) => item.providerResourceId === "svc-2");
    expect(missing?.discoveryStatus).toBe("missing");
    expect(missing?.missingSince).toBeTruthy();

    await stack.discovery.sync({
      connectionId: a.id,
      installedPluginId: installed!.id,
      pluginId: "example.local",
      items: [
        {
          providerResourceId: "svc-2",
          resourceType: "local.service",
          name: "Two",
          metadata: {},
          pluginVersion: "1.0.0",
          schemaVersion: "1",
        },
      ],
    });
    const rediscovered = await stack.db.discoveredResources.getById(
      missing!.id,
    );
    expect(rediscovered?.discoveryStatus).toBe("active");
    expect(rediscovered?.missingSince).toBeUndefined();

    await stack.discovery.sync({
      connectionId: b.id,
      installedPluginId: installed!.id,
      pluginId: "example.local",
      items: [
        {
          providerResourceId: "svc-1",
          resourceType: "local.service",
          name: "Other connection",
          metadata: {},
          pluginVersion: "1.0.0",
          schemaVersion: "1",
        },
      ],
    });
    const fromB = await stack.discovery.listByConnectionId(b.id);
    expect(fromB[0]?.id).not.toBe(first[0]?.id);

    await expect(
      stack.discovery.sync({
        connectionId: a.id,
        installedPluginId: installed!.id,
        pluginId: "example.local",
        items: [
          {
            providerResourceId: "dup",
            resourceType: "local.service",
            name: "Dup",
            metadata: {},
            pluginVersion: "1.0.0",
            schemaVersion: "1",
          },
          {
            providerResourceId: "dup",
            resourceType: "local.service",
            name: "Dup2",
            metadata: {},
            pluginVersion: "1.0.0",
            schemaVersion: "1",
          },
        ],
      }),
    ).rejects.toThrow(/Duplicate/);
  });
});

describe("Bindings", () => {
  it("binds, rejects env/project mismatch, detaches without delete", async () => {
    const stack = createStack();
    const [installed] = await stack.installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    const connection = await stack.connections.create({
      installedPluginId: installed!.id,
      name: "Conn",
      status: "connected",
    });
    const [resource] = await stack.discovery.sync({
      connectionId: connection.id,
      installedPluginId: installed!.id,
      pluginId: "example.local",
      items: [
        {
          providerResourceId: "svc-1",
          resourceType: "local.service",
          name: "API",
          metadata: {},
          pluginVersion: "1.0.0",
          schemaVersion: "1",
        },
      ],
    });

    const bound = await stack.bindings.bind({
      projectId: "project-a",
      environmentId: "env-a",
      expectedProjectIdForEnvironment: "project-a",
      discoveredResourceId: resource!.id,
      createdBy: actor,
    });
    expect(bound.bindingStatus).toBe("active");

    await expect(
      stack.bindings.bind({
        projectId: "project-a",
        environmentId: "env-other",
        expectedProjectIdForEnvironment: "project-b",
        discoveredResourceId: resource!.id,
        createdBy: actor,
      }),
    ).rejects.toBeInstanceOf(PluginDomainError);

    const suggested = await stack.bindings.bind({
      projectId: "project-a",
      discoveredResourceId: resource!.id,
      bindingStatus: "suggested",
      createdBy: actor,
    });
    expect(suggested.bindingStatus).toBe("suggested");

    const detached = await stack.bindings.detach(bound.id);
    expect(detached.bindingStatus).toBe("detached");
    expect(await stack.db.resourceBindings.getById(bound.id)).toBeTruthy();
  });
});

describe("State", () => {
  it("keeps observed and desired separate with optimistic concurrency", async () => {
    const stack = createStack();
    const [installed] = await stack.installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    const connection = await stack.connections.create({
      installedPluginId: installed!.id,
      name: "Conn",
      status: "connected",
    });
    const [resource] = await stack.discovery.sync({
      connectionId: connection.id,
      installedPluginId: installed!.id,
      pluginId: "example.local",
      items: [
        {
          providerResourceId: "svc-1",
          resourceType: "local.service",
          name: "API",
          metadata: {},
          pluginVersion: "1.0.0",
          schemaVersion: "1",
        },
      ],
    });
    const binding = await stack.bindings.bind({
      projectId: "project-a",
      discoveredResourceId: resource!.id,
      createdBy: actor,
    });

    const observed = await stack.state.recordObserved({
      discoveredResourceId: resource!.id,
      pluginId: "example.local",
      connectionId: connection.id,
      state: { port: 3000, secretPresent: true },
      pluginVersion: "1.0.0",
      schemaVersion: "1",
      observedAt: "2026-01-01T00:00:00.000Z",
      checksum: "abc",
    });
    await stack.state.recordObserved({
      ...observed,
      state: { port: 3001, secretPresent: true },
      observedAt: "2026-01-01T00:01:00.000Z",
      checksum: "def",
    });
    expect(
      (await stack.state.getObserved(resource!.id))?.state.port,
    ).toBe(3001);
    expect(await stack.state.listObservedHistory(resource!.id)).toHaveLength(2);

    const desired = await stack.state.saveDesired({
      projectId: "project-a",
      resourceBindingId: binding.id,
      pluginId: "example.local",
      connectionId: connection.id,
      state: { port: 4000 },
      schemaVersion: "1",
      createdBy: actor,
    });
    expect(desired.revision).toBe(1);

    const next = await stack.state.saveDesired({
      projectId: "project-a",
      resourceBindingId: binding.id,
      pluginId: "example.local",
      connectionId: connection.id,
      state: { port: 4001 },
      schemaVersion: "1",
      createdBy: actor,
      expectedRevision: 1,
    });
    expect(next.revision).toBe(2);

    await expect(
      stack.state.saveDesired({
        projectId: "project-a",
        resourceBindingId: binding.id,
        pluginId: "example.local",
        connectionId: connection.id,
        state: { port: 4002 },
        schemaVersion: "1",
        createdBy: actor,
        expectedRevision: 1,
      }),
    ).rejects.toBeInstanceOf(OptimisticConcurrencyError);

    await expect(
      stack.state.saveDesired({
        projectId: "project-a",
        resourceBindingId: binding.id,
        pluginId: "example.local",
        connectionId: connection.id,
        state: { apiToken: "plaintext-secret" },
        schemaVersion: "1",
        createdBy: actor,
        expectedRevision: 2,
      }),
    ).rejects.toBeInstanceOf(PluginDomainError);

    await expect(
      stack.state.saveDesired({
        projectId: "project-a",
        resourceBindingId: binding.id,
        pluginId: "example.local",
        connectionId: connection.id,
        state: { credentials: { apiToken: "nested-secret" } },
        schemaVersion: "1",
        createdBy: actor,
        expectedRevision: 2,
      }),
    ).rejects.toBeInstanceOf(PluginDomainError);

    await expect(
      stack.state.recordObserved({
        discoveredResourceId: resource!.id,
        pluginId: "example.local",
        connectionId: connection.id,
        state: { password: "should-not-store" },
        pluginVersion: "1.0.0",
        schemaVersion: "1",
        observedAt: "2026-01-01T00:02:00.000Z",
      }),
    ).rejects.toBeInstanceOf(PluginDomainError);

    expect((await stack.state.getObserved(resource!.id))?.state.port).toBe(3001);
    expect((await stack.state.getDesired(binding.id))?.state.port).toBe(4001);
  });
});

describe("Plans and approvals", () => {
  it("persists immutable plans, approvals, rejections, and apply gating", async () => {
    const stack = createStack();
    const [installed] = await stack.installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    const connection = await stack.connections.create({
      installedPluginId: installed!.id,
      name: "Conn",
      status: "connected",
    });
    const [resource] = await stack.discovery.sync({
      connectionId: connection.id,
      installedPluginId: installed!.id,
      pluginId: "example.local",
      items: [
        {
          providerResourceId: "svc-1",
          resourceType: "local.service",
          name: "API",
          metadata: {},
          pluginVersion: "1.0.0",
          schemaVersion: "1",
        },
      ],
    });
    const binding = await stack.bindings.bind({
      projectId: "project-a",
      discoveredResourceId: resource!.id,
      createdBy: actor,
    });

    const planBody: ChangePlan = {
      id: "plan-1",
      pluginId: "example.local",
      resourceId: resource!.id,
      summary: "Update port",
      operations: [
        {
          id: "op-1",
          type: "set_port",
          description: "Set port",
          requiresApproval: true,
          destructive: true,
        },
      ],
      warnings: [],
      destructive: true,
    };

    const plan = await stack.plans.create({
      pluginId: "example.local",
      connectionId: connection.id,
      projectId: "project-a",
      resourceBindingId: binding.id,
      plan: planBody,
      createdBy: actor,
    });
    expect(plan.planSchemaVersion).toBe("1");

    await expect(
      stack.db.changePlans.save({ ...plan, status: "approved" }),
    ).rejects.toThrow(/immutable/);

    await expect(
      stack.approvals.approve({
        changePlanId: plan.id,
        approvedOperationIds: ["op-1"],
        destructiveApproval: false,
        approvedBy: actor,
      }),
    ).rejects.toBeInstanceOf(PluginDomainError);

    await expect(stack.approvals.beginApply(plan.id)).rejects.toBeInstanceOf(
      PluginDomainError,
    );

    const approval = await stack.approvals.approve({
      changePlanId: plan.id,
      approvedOperationIds: ["op-1"],
      destructiveApproval: true,
      approvedBy: actor,
      comment: "looks good",
    });
    expect(approval.destructiveApproval).toBe(true);

    const approved = await stack.approvals.buildApprovedChangePlan(plan.id);
    expect(approved.approvalId).toBe(approval.id);
    expect(approved.approvedOperationIds).toEqual(["op-1"]);

    // Supersede an approved (not yet applying) plan is allowed.
    const plan2 = await stack.plans.create({
      pluginId: "example.local",
      connectionId: connection.id,
      projectId: "project-a",
      resourceBindingId: binding.id,
      plan: { ...planBody, id: "plan-2" },
      createdBy: actor,
      supersedePlanId: plan.id,
    });
    expect((await stack.plans.getById(plan.id))?.status).toBe("superseded");
    expect(plan2.status).toBe("pending");

    await stack.db.changePlans.setStatus(plan2.id, "applying");
    await expect(
      stack.plans.create({
        pluginId: "example.local",
        connectionId: connection.id,
        projectId: "project-a",
        resourceBindingId: binding.id,
        plan: { ...planBody, id: "plan-3" },
        createdBy: actor,
        supersedePlanId: plan2.id,
      }),
    ).rejects.toBeInstanceOf(PluginDomainError);

    await stack.db.changePlans.setStatus(plan2.id, "pending");
    await stack.approvals.reject({
      changePlanId: plan2.id,
      rejectedBy: actor,
      reason: "not now",
    });
    expect((await stack.plans.getById(plan2.id))?.status).toBe("rejected");
  });
});

describe("Apply and verify", () => {
  it("records apply and separate verification with execution history linkage", async () => {
    const stack = createStack();
    const [installed] = await stack.installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    const connection = await stack.connections.create({
      installedPluginId: installed!.id,
      name: "Conn",
      status: "connected",
    });
    const [resource] = await stack.discovery.sync({
      connectionId: connection.id,
      installedPluginId: installed!.id,
      pluginId: "example.local",
      items: [
        {
          providerResourceId: "svc-1",
          resourceType: "local.service",
          name: "API",
          metadata: {},
          pluginVersion: "1.0.0",
          schemaVersion: "1",
        },
      ],
    });
    const binding = await stack.bindings.bind({
      projectId: "project-a",
      discoveredResourceId: resource!.id,
      createdBy: actor,
    });
    const plan = await stack.plans.create({
      pluginId: "example.local",
      connectionId: connection.id,
      projectId: "project-a",
      resourceBindingId: binding.id,
      plan: {
        id: "plan-apply",
        pluginId: "example.local",
        resourceId: resource!.id,
        summary: "Apply",
        operations: [
          {
            id: "op-1",
            type: "set",
            description: "set",
            requiresApproval: true,
          },
        ],
        warnings: [],
        destructive: false,
      },
      createdBy: actor,
    });
    await stack.approvals.approve({
      changePlanId: plan.id,
      approvedOperationIds: ["op-1"],
      destructiveApproval: false,
      approvedBy: actor,
    });

    await stack.approvals.beginApply(plan.id);
    const apply = await stack.approvals.completeApply({
      changePlanId: plan.id,
      executionId: "exec-apply",
      status: "succeeded",
      startedAt: "2026-01-01T00:00:00.000Z",
      result: {
        ok: true,
        message: "applied",
        appliedOperationIds: ["op-1"],
      },
    });
    expect(apply.status).toBe("succeeded");
    expect((await stack.plans.getById(plan.id))?.status).toBe("applied");

    const verification = await stack.approvals.recordVerification({
      changeApplyId: apply.id,
      executionId: "exec-verify",
      status: "failed",
      result: {
        ok: false,
        message: "drift",
        mismatches: ["port"],
      },
    });
    expect(verification.status).toBe("failed");
    expect(verification.changeApplyId).toBe(apply.id);

    const sink = createPersistentExecutionEventSink({
      history: stack.db.executionHistory,
      connectionId: connection.id,
    });
    await sink.record({
      executionId: "exec-apply",
      pluginId: "example.local",
      pluginVersion: "1.0.0",
      capability: "apply",
      status: "succeeded",
      startedAt: "2026-01-01T00:00:00.000Z",
      finishedAt: "2026-01-01T00:00:01.000Z",
      durationMs: 1000,
      actor,
      warningCount: 0,
    });
    const history = await stack.db.executionHistory.getByExecutionId(
      "exec-apply",
    );
    expect(history?.connectionId).toBe(connection.id);
    expect(JSON.stringify(history)).not.toMatch(/token|password|secret/i);

    const failedPlan = await stack.plans.create({
      pluginId: "example.local",
      connectionId: connection.id,
      projectId: "project-a",
      resourceBindingId: binding.id,
      plan: {
        id: "plan-fail",
        pluginId: "example.local",
        resourceId: resource!.id,
        summary: "Fail",
        operations: [
          {
            id: "op-1",
            type: "set",
            description: "set",
            requiresApproval: false,
          },
        ],
        warnings: [],
        destructive: false,
      },
      createdBy: actor,
    });
    await stack.approvals.approve({
      changePlanId: failedPlan.id,
      approvedOperationIds: ["op-1"],
      destructiveApproval: false,
      approvedBy: actor,
    });
    await stack.approvals.beginApply(failedPlan.id);
    const failedApply = await stack.approvals.completeApply({
      changePlanId: failedPlan.id,
      executionId: "exec-fail",
      status: "failed",
      startedAt: "2026-01-01T00:00:00.000Z",
      error: {
        code: "execution_failed",
        message: "boom",
        pluginId: "example.local",
        capability: "apply",
        retryable: true,
      },
    });
    expect(failedApply.status).toBe("failed");
  });
});

describe("Deletion and disconnection", () => {
  it("disable and missing preserve records while blocking execution", async () => {
    const stack = createStack();
    const [installed] = await stack.installation.reconcileBuiltIns([
      { manifest: manifest() },
    ]);
    const connection = await stack.connections.create({
      installedPluginId: installed!.id,
      name: "Conn",
      status: "connected",
    });
    await stack.db.executionHistory.append({
      id: "h1",
      executionId: "e1",
      pluginId: "example.local",
      pluginVersion: "1.0.0",
      capability: "discover",
      status: "succeeded",
      startedAt: "2026-01-01T00:00:00.000Z",
      finishedAt: "2026-01-01T00:00:01.000Z",
      durationMs: 1,
      actor,
      connectionId: connection.id,
      warningCount: 0,
      recordedAt: "2026-01-01T00:00:01.000Z",
    });

    await stack.installation.disable(installed!.id);
    expect(
      (await stack.guard.assertExecutable({ pluginId: "example.local" })).ok,
    ).toBe(false);

    await stack.installation.reconcileBuiltIns([]);
    expect(await stack.db.connections.getById(connection.id)).toBeTruthy();
    expect(await stack.db.executionHistory.getByExecutionId("e1")).toBeTruthy();

    const resolver = createPersistentPermissionResolver({
      connections: stack.db.connections,
      grants: stack.db.permissionGrants,
      connectionId: connection.id,
    });
    expect(
      await resolver.resolve({
        pluginId: "example.local",
        capability: "discover",
        actor,
      }),
    ).toEqual([]);
  });
});

describe("Environment mapping", () => {
  it("stores suggestions without auto-binding", async () => {
    const stack = createStack();
    const suggestion = await stack.mapping.createSuggestion({
      projectId: "project-a",
      connectionId: "conn",
      discoveredResourceId: "res",
      suggestedEnvironmentName: "production",
      confidence: 0.8,
      reasons: ["name contains production"],
    });
    expect(suggestion.status).toBe("pending");
    const accepted = await stack.mapping.accept({
      suggestionId: suggestion.id,
      resolvedBy: actor,
    });
    expect(accepted.status).toBe("accepted");
    expect(await stack.db.resourceBindings.listByProjectId("project-a")).toEqual(
      [],
    );
  });
});
