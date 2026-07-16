import { describe, expect, it, vi } from "vitest";
import { RAYVAN_PLUGIN_API_VERSION } from "../../src/api-version.js";
import type {
  ApplyResult,
  ChangePlan,
  DiscoveredResource,
  ObservedResourceState,
  VerificationResult,
} from "../../src/contracts/index.js";
import { PluginExecutionError } from "../../src/errors/index.js";
import type { PluginManifest } from "../../src/manifest/index.js";
import type { RayvanPlugin } from "../../src/plugin.js";
import {
  createPluginExecutionStack,
  InMemoryPluginExecutionEventSink,
  InMemoryPluginPermissionResolver,
  PluginExecutionService,
  redactSecrets,
  type PluginCapabilityPermissionPolicy,
  type PluginExecutionActor,
  type PluginRuntime,
  type PluginRuntimeInvocation,
} from "../../src/execution/index.js";

const actor: PluginExecutionActor = {
  id: "user-1",
  type: "user",
  displayName: "Test User",
};

function baseManifest(
  overrides: Partial<PluginManifest> & Pick<PluginManifest, "id" | "capabilities">,
): PluginManifest {
  return {
    name: overrides.id,
    version: "0.1.0",
    publisher: "rayvan",
    rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
    permissions: overrides.permissions ?? [],
    resourceTypes: [
      {
        id: "test.resource",
        name: "Test Resource",
        schemaVersion: "1.0.0",
      },
    ],
    ...overrides,
  };
}

function observedState(
  resourceId = "res-1",
  pluginId = "demo",
): ObservedResourceState {
  return {
    resourceId,
    pluginId,
    resourceType: "test.resource",
    observedAt: "1970-01-01T00:00:00.000Z",
    status: "ready",
    attributes: { port: 3000 },
  };
}

function changePlanFixture(overrides?: Partial<ChangePlan>): ChangePlan {
  return {
    id: "plan-1",
    pluginId: "demo",
    resourceId: "res-1",
    summary: "Update port",
    operations: [
      {
        id: "op-1",
        type: "update_attribute",
        description: "Set port",
        path: "attributes.port",
        before: 3000,
        after: 4000,
        requiresApproval: true,
        destructive: false,
      },
    ],
    warnings: [],
    destructive: false,
    ...overrides,
  };
}

function createDemoPlugin(options?: {
  permissions?: PluginManifest["permissions"];
  discover?: RayvanPlugin["discover"];
  inspect?: RayvanPlugin["inspect"];
  plan?: RayvanPlugin["plan"];
  apply?: RayvanPlugin["apply"];
  verify?: RayvanPlugin["verify"];
  evaluateFindings?: RayvanPlugin["evaluateFindings"];
  capabilities?: PluginManifest["capabilities"];
  hangMs?: number;
}): RayvanPlugin {
  const hangMs = options?.hangMs;
  const wait = async (signal?: AbortSignal) => {
    if (!hangMs) {
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(resolve, hangMs);
      signal?.addEventListener(
        "abort",
        () => {
          clearTimeout(timer);
          reject(signal.reason ?? new Error("aborted"));
        },
        { once: true },
      );
    });
  };

  const capabilities =
    options?.capabilities ??
    ([
      "discover",
      "inspect",
      "plan",
      "apply",
      "verify",
      ...(options?.evaluateFindings ? (["evaluate_findings"] as const) : []),
    ] as PluginManifest["capabilities"]);

  return {
    manifest: baseManifest({
      id: "demo",
      capabilities,
      permissions: options?.permissions ?? [],
    }),
    discover:
      options?.discover ??
      (capabilities.includes("discover")
        ? async () => {
            await wait();
            return [
              {
                providerResourceId: "svc-1",
                resourceType: "test.resource",
                name: "API",
                metadata: {},
                schemaVersion: "1.0.0",
              },
            ] satisfies DiscoveredResource[];
          }
        : undefined),
    inspect:
      options?.inspect ??
      (capabilities.includes("inspect")
        ? async () => {
            await wait();
            return observedState();
          }
        : undefined),
    plan:
      options?.plan ??
      (capabilities.includes("plan")
        ? async () => {
            await wait();
            return changePlanFixture();
          }
        : undefined),
    apply:
      options?.apply ??
      (capabilities.includes("apply")
        ? async () => {
            await wait();
            return {
              ok: true,
              message: "Applied",
              appliedOperationIds: ["op-1"],
              resultingState: observedState(),
            } satisfies ApplyResult;
          }
        : undefined),
    verify:
      options?.verify ??
      (capabilities.includes("verify")
        ? async () => {
            await wait();
            return {
              ok: true,
              message: "Verified",
              observed: observedState(),
              mismatches: [],
            } satisfies VerificationResult;
          }
        : undefined),
    evaluateFindings: options?.evaluateFindings,
  };
}

describe("PluginExecutionService", () => {
  it("runs discover, inspect, plan, approved apply, and verify successfully", async () => {
    let clock = 1_000;
    const { executionService } = createPluginExecutionStack({
      plugins: [createDemoPlugin()],
      idFactory: () => "exec-1",
      now: () => new Date(clock),
    });

    const discover = await executionService.discover({
      pluginId: "demo",
      actor,
      context: { pluginId: "demo", integrationId: "int-1" },
    });
    expect(discover.status).toBe("succeeded");
    if (discover.status === "succeeded") {
      expect(discover.data).toHaveLength(1);
      expect(discover.pluginVersion).toBe("0.1.0");
      expect(discover.executionId).toBe("exec-1");
      expect(discover.durationMs).toBe(0);
    }

    clock = 1_050;
    const inspect = await executionService.inspect({
      pluginId: "demo",
      actor,
      resourceId: "res-1",
      context: {
        pluginId: "demo",
        integrationId: "int-1",
        resource: {
          resourceId: "res-1",
          pluginId: "demo",
          providerResourceId: "svc-1",
          resourceType: "test.resource",
        },
      },
    });
    expect(inspect.status).toBe("succeeded");

    clock = 1_100;
    const plan = await executionService.plan({
      pluginId: "demo",
      actor,
      resourceId: "res-1",
      context: {
        pluginId: "demo",
        integrationId: "int-1",
        resource: {
          resourceId: "res-1",
          pluginId: "demo",
          providerResourceId: "svc-1",
          resourceType: "test.resource",
        },
        observed: observedState(),
        desired: {
          resourceId: "res-1",
          pluginId: "demo",
          resourceType: "test.resource",
          attributes: { port: 4000 },
        },
      },
    });
    expect(plan.status).toBe("succeeded");
    if (plan.status !== "succeeded") {
      throw new Error("expected plan success");
    }

    clock = 1_200;
    const apply = await executionService.apply({
      pluginId: "demo",
      actor,
      resourceId: "res-1",
      context: {
        pluginId: "demo",
        integrationId: "int-1",
        resource: {
          resourceId: "res-1",
          pluginId: "demo",
          providerResourceId: "svc-1",
          resourceType: "test.resource",
        },
        approvedPlan: {
          plan: plan.data,
          approvalId: "approval-1",
          approvedAt: "1970-01-01T00:00:00.000Z",
          approvedOperationIds: ["op-1"],
        },
      },
    });
    expect(apply.status).toBe("succeeded");
    if (apply.status !== "succeeded") {
      throw new Error("expected apply success");
    }

    clock = 1_300;
    const verify = await executionService.verify({
      pluginId: "demo",
      actor,
      resourceId: "res-1",
      context: {
        pluginId: "demo",
        integrationId: "int-1",
        resource: {
          resourceId: "res-1",
          pluginId: "demo",
          providerResourceId: "svc-1",
          resourceType: "test.resource",
        },
        approvedPlan: {
          plan: plan.data,
          approvalId: "approval-1",
          approvedAt: "1970-01-01T00:00:00.000Z",
          approvedOperationIds: ["op-1"],
        },
        applyResult: apply.data,
      },
    });
    expect(verify.status).toBe("succeeded");
    expect(verify.startedAt).toBe(new Date(1_300).toISOString());
    expect(verify.finishedAt).toBe(new Date(1_300).toISOString());
  });

  it("runs evaluate_findings and validates detections", async () => {
    const { executionService } = createPluginExecutionStack({
      plugins: [
        createDemoPlugin({
          capabilities: ["evaluate_findings"],
          evaluateFindings: async () => ({
            detections: [
              {
                ruleId: "demo.port-ready",
                severity: "info",
                title: "Service ready",
                summary: "Observed resource is ready",
                scope: { resourceBindingId: "res-1" },
                evidence: [
                  {
                    type: "resource_state",
                    resourceBindingId: "res-1",
                    state: "ready",
                  },
                ],
                fingerprintParts: ["demo", "port-ready", "res-1"],
              },
            ],
            warnings: [],
          }),
        }),
      ],
      idFactory: () => "exec-findings",
    });

    const result = await executionService.evaluateFindings({
      pluginId: "demo",
      projectId: "project-1",
      actor,
      context: {
        pluginId: "demo",
        projectId: "project-1",
        connectionId: "conn-1",
        environments: [{ id: "env-1" }],
        resources: [{ resourceBindingId: "res-1" }],
        observedStates: [],
      },
    });

    expect(result.status).toBe("succeeded");
    if (result.status === "succeeded") {
      expect(result.data.detections).toHaveLength(1);
      expect(result.data.detections[0]?.ruleId).toBe("demo.port-ready");
    }
  });

  it("rejects evaluate_findings output with secret evidence", async () => {
    const { executionService } = createPluginExecutionStack({
      plugins: [
        createDemoPlugin({
          capabilities: ["evaluate_findings"],
          evaluateFindings: async () => ({
            detections: [
              {
                ruleId: "demo.leaky",
                title: "Leak",
                summary: "Bad evidence",
                scope: {},
                evidence: [
                  {
                    type: "message",
                    message: "api_key=sk-live-abcdefghijklmnopqrstuvwxyz",
                  },
                ],
                fingerprintParts: ["demo", "leaky"],
              },
            ],
            warnings: [],
          }),
        }),
      ],
    });

    const result = await executionService.evaluateFindings({
      pluginId: "demo",
      actor,
      context: {
        pluginId: "demo",
        projectId: "project-1",
        connectionId: "conn-1",
        environments: [],
        resources: [],
        observedStates: [],
      },
    });

    expect(result.status).toBe("failed");
    expect(result.error?.code).toBe("validation_failed");
  });

  it("returns not_found for a missing plugin", async () => {
    const { executionService } = createPluginExecutionStack();
    const result = await executionService.discover({
      pluginId: "missing",
      actor,
      context: { pluginId: "missing", integrationId: "int-1" },
    });
    expect(result.status).toBe("failed");
    expect(result.error?.code).toBe("not_found");
  });

  it("returns capability_unsupported when undeclared", async () => {
    const plugin: RayvanPlugin = {
      manifest: baseManifest({
        id: "auth-only",
        capabilities: ["authenticate"],
      }),
      authenticate: async () => ({ ok: true, message: "ok" }),
    };
    const { executionService } = createPluginExecutionStack({ plugins: [plugin] });
    const result = await executionService.discover({
      pluginId: "auth-only",
      actor,
      context: { pluginId: "auth-only", integrationId: "int-1" },
    });
    expect(result.status).toBe("failed");
    expect(result.error?.code).toBe("capability_unsupported");
  });

  it("returns missing_handler when declared capability has no handler", async () => {
    const plugin = {
      manifest: baseManifest({
        id: "broken",
        capabilities: ["discover"],
      }),
      discover: async () => [],
    } as RayvanPlugin;
    // Bypass registry validation by constructing service with a stub registry.
    const registry = {
      register() {},
      unregister() {},
      get: () => {
        const broken = { ...plugin };
        delete (broken as { discover?: unknown }).discover;
        return broken;
      },
      list: () => [plugin.manifest],
      supports: () => true,
    };
    const runtime: PluginRuntime = {
      invoke: async () => {
        throw new Error("should not be called");
      },
    };
    const service = new PluginExecutionService({
      registry,
      runtime,
      permissionResolver: new InMemoryPluginPermissionResolver(),
    });
    const result = await service.discover({
      pluginId: "broken",
      actor,
      context: { pluginId: "broken", integrationId: "int-1" },
    });
    expect(result.status).toBe("failed");
    expect(result.error?.code).toBe("missing_handler");
  });

  it("denies when capability policy requires a permission that is not granted", async () => {
    const capabilityPermissions: PluginCapabilityPermissionPolicy = {
      authenticate: [],
      discover: ["network"],
      inspect: [],
      plan: [],
      apply: [],
      verify: [],
    };
    const { executionService } = createPluginExecutionStack({
      plugins: [
        createDemoPlugin({
          permissions: ["network"],
        }),
      ],
      permissionResolver: new InMemoryPluginPermissionResolver(),
      capabilityPermissions,
    });
    const result = await executionService.discover({
      pluginId: "demo",
      actor,
      context: { pluginId: "demo", integrationId: "int-1" },
    });
    expect(result.status).toBe("failed");
    expect(result.error?.code).toBe("permission_denied");
  });

  it("rejects invalid execution requests and invalid plugin output", async () => {
    const { executionService } = createPluginExecutionStack({
      plugins: [
        createDemoPlugin({
          discover: async () =>
            [
              {
                providerResourceId: "",
                resourceType: "test.resource",
                name: "bad",
                metadata: {},
                schemaVersion: "1.0.0",
              },
            ] as DiscoveredResource[],
        }),
      ],
    });

    const invalidRequest = await executionService.discover({
      pluginId: "demo",
      actor: { id: "", type: "user" },
      context: { pluginId: "demo", integrationId: "int-1" },
    });
    expect(invalidRequest.status).toBe("failed");
    expect(invalidRequest.error?.code).toBe("validation_failed");

    const invalidOutput = await executionService.discover({
      pluginId: "demo",
      actor,
      context: { pluginId: "demo", integrationId: "int-1" },
    });
    expect(invalidOutput.status).toBe("failed");
    expect(invalidOutput.error?.code).toBe("validation_failed");
  });

  it("normalizes typed errors, strings, and arbitrary thrown objects", async () => {
    const cases: Array<{
      discover: RayvanPlugin["discover"];
      messageIncludes: string;
    }> = [
      {
        discover: async () => {
          throw new PluginExecutionError("demo", "discover", "typed boom");
        },
        messageIncludes: "typed boom",
      },
      {
        discover: async () => {
          throw "string boom";
        },
        messageIncludes: "string boom",
      },
      {
        discover: async () => {
          throw { weird: true, token: "secret-value" };
        },
        messageIncludes: "non-error",
      },
    ];

    for (const testCase of cases) {
      const { executionService } = createPluginExecutionStack({
        plugins: [createDemoPlugin({ discover: testCase.discover })],
      });
      const result = await executionService.discover({
        pluginId: "demo",
        actor,
        context: { pluginId: "demo", integrationId: "int-1" },
      });
      expect(result.status).toBe("failed");
      expect(result.error?.code).toBe("execution_failed");
      expect(result.error?.message).toContain(testCase.messageIncludes);
      if (testCase.messageIncludes === "non-error") {
        expect(JSON.stringify(result.error?.details)).not.toContain(
          "secret-value",
        );
      }
    }
  });

  it("times out and cancels without brittle sleeps", async () => {
    const { executionService } = createPluginExecutionStack({
      plugins: [createDemoPlugin({ hangMs: 100 })],
    });

    const timedOut = await executionService.discover({
      pluginId: "demo",
      actor,
      timeoutMs: 5,
      context: { pluginId: "demo", integrationId: "int-1" },
    });
    expect(timedOut.status).toBe("timed_out");
    expect(timedOut.error?.code).toBe("timeout");

    const controller = new AbortController();
    controller.abort(new Error("user cancel"));
    const cancelled = await executionService.discover({
      pluginId: "demo",
      actor,
      signal: controller.signal,
      context: { pluginId: "demo", integrationId: "int-1" },
    });
    expect(cancelled.status).toBe("cancelled");
    expect(cancelled.error?.code).toBe("cancelled");
  });

  it("rejects unapproved, partially approved, and destructive applies", async () => {
    const { executionService } = createPluginExecutionStack({
      plugins: [createDemoPlugin()],
    });

    const resource = {
      resourceId: "res-1",
      pluginId: "demo",
      providerResourceId: "svc-1",
      resourceType: "test.resource",
    };

    const unapproved = await executionService.apply({
      pluginId: "demo",
      actor,
      resourceId: "res-1",
      context: {
        pluginId: "demo",
        integrationId: "int-1",
        resource,
        approvedPlan: {
          plan: changePlanFixture(),
          approvalId: "a1",
          approvedAt: "1970-01-01T00:00:00.000Z",
          approvedOperationIds: [],
        },
      },
    });
    expect(unapproved.status).toBe("failed");
    expect(unapproved.error?.code).toBe("approval_invalid");

    const partial = await executionService.apply({
      pluginId: "demo",
      actor,
      resourceId: "res-1",
      context: {
        pluginId: "demo",
        integrationId: "int-1",
        resource,
        approvedPlan: {
          plan: changePlanFixture({
            operations: [
              {
                id: "op-1",
                type: "update_attribute",
                description: "one",
                requiresApproval: true,
              },
              {
                id: "op-2",
                type: "update_attribute",
                description: "two",
                requiresApproval: true,
              },
            ],
          }),
          approvalId: "a1",
          approvedAt: "1970-01-01T00:00:00.000Z",
          approvedOperationIds: ["op-1"],
        },
      },
    });
    expect(partial.status).toBe("failed");
    expect(partial.error?.code).toBe("approval_invalid");

    const destructive = await executionService.apply({
      pluginId: "demo",
      actor,
      resourceId: "res-1",
      context: {
        pluginId: "demo",
        integrationId: "int-1",
        resource,
        approvedPlan: {
          plan: changePlanFixture({
            destructive: true,
            operations: [
              {
                id: "op-1",
                type: "delete",
                description: "delete",
                requiresApproval: true,
                destructive: true,
              },
            ],
          }),
          approvalId: "a1",
          approvedAt: "1970-01-01T00:00:00.000Z",
          approvedOperationIds: ["op-1"],
        },
      },
    });
    expect(destructive.status).toBe("failed");
    expect(destructive.error?.code).toBe("approval_invalid");
  });

  it("emits events on success and failure, and sink failure becomes a warning", async () => {
    const sink = new InMemoryPluginExecutionEventSink();
    const { executionService } = createPluginExecutionStack({
      plugins: [createDemoPlugin()],
      eventSink: sink,
    });

    const ok = await executionService.discover({
      pluginId: "demo",
      actor,
      context: { pluginId: "demo", integrationId: "int-1" },
    });
    expect(ok.status).toBe("succeeded");
    expect(sink.events).toHaveLength(1);
    expect(sink.events[0]?.status).toBe("succeeded");

    const failingSink = {
      record: async () => {
        throw new Error("sink down");
      },
    };
    const stack = createPluginExecutionStack({
      plugins: [createDemoPlugin()],
      eventSink: failingSink,
    });
    const stillOk = await stack.executionService.discover({
      pluginId: "demo",
      actor,
      context: { pluginId: "demo", integrationId: "int-1" },
    });
    expect(stillOk.status).toBe("succeeded");
    expect(stillOk.warnings.some((warning) => warning.includes("sink down"))).toBe(
      true,
    );

    const fail = await executionService.discover({
      pluginId: "missing",
      actor,
      context: { pluginId: "missing", integrationId: "int-1" },
    });
    expect(fail.status).toBe("failed");
    expect(sink.events.at(-1)?.status).toBe("failed");
  });

  it("redacts secrets in nested payloads and tolerates cycles", () => {
    const redacted = redactSecrets({
      token: "abc",
      nested: { apiKey: "xyz", safe: "ok" },
      Authorization: "Bearer x",
    });
    expect(redacted).toEqual({
      token: "[REDACTED]",
      nested: { apiKey: "[REDACTED]", safe: "ok" },
      Authorization: "[REDACTED]",
    });

    const cyclic: Record<string, unknown> = { password: "p", safe: "ok" };
    cyclic.self = cyclic;
    expect(redactSecrets(cyclic)).toEqual({
      password: "[REDACTED]",
      safe: "ok",
      self: "[Circular]",
    });
  });

  it("invokes through PluginRuntime rather than handlers directly", async () => {
    const plugin = createDemoPlugin();
    const invoke = vi.fn(
      async <TInput, TOutput>(
        invocation: PluginRuntimeInvocation<TInput>,
      ): Promise<TOutput> => {
        expect(invocation.pluginId).toBe("demo");
        expect(invocation.capability).toBe("discover");
        return [
          {
            providerResourceId: "svc-1",
            resourceType: "test.resource",
            name: "API",
            metadata: {},
            schemaVersion: "1.0.0",
          },
        ] as TOutput;
      },
    );
    const runtime: PluginRuntime = { invoke };

    const { executionService } = createPluginExecutionStack({
      plugins: [plugin],
      runtime,
    });

    const spyDiscover = vi.fn(plugin.discover!);
    plugin.discover = spyDiscover;

    const result = await executionService.discover({
      pluginId: "demo",
      actor,
      context: { pluginId: "demo", integrationId: "int-1" },
    });

    expect(result.status).toBe("succeeded");
    expect(invoke).toHaveBeenCalledTimes(1);
    expect(spyDiscover).not.toHaveBeenCalled();
  });
});
