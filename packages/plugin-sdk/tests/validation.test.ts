import { describe, expect, it } from "vitest";
import {
  PluginValidationError,
  validateApplyResult,
  validateApprovedChangePlan,
  validateChangePlan,
  validateDiscoveredResource,
  validateEvaluateFindingsContext,
  validateEvaluateFindingsResult,
  validateObservedResourceState,
  validatePluginFindingDetection,
  validatePluginManifest,
  validatePluginResource,
  validateVerificationResult,
  RAYVAN_PLUGIN_API_VERSION,
  type ApplyResult,
  type ChangePlan,
  type DiscoveredResource,
  type EvaluateFindingsContext,
  type EvaluateFindingsResult,
  type ObservedResourceState,
  type PluginFindingDetection,
  type PluginManifest,
  type PluginResource,
  type VerificationResult,
} from "../src/index.js";

const validManifest: PluginManifest = {
  id: "example-local",
  name: "Example Local",
  description: "Demo plugin",
  version: "0.1.0",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: ["discover", "inspect"],
  permissions: [],
  resourceTypes: [
    {
      id: "local.service",
      name: "Local Service",
      schemaVersion: "1.0.0",
    },
  ],
};

function validDetection(
  overrides: Partial<PluginFindingDetection> = {},
): PluginFindingDetection {
  return {
    ruleId: "example-local.port-check",
    severity: "info",
    title: "Port configured",
    summary: "Local service has a port",
    scope: { environmentId: "env-1" },
    evidence: [{ type: "message", message: "Port 3000 is set" }],
    fingerprintParts: ["example-local", "port-check", "env-1"],
    ...overrides,
  };
}

describe("plugin model validation", () => {
  it("accepts a valid manifest", () => {
    expect(() => validatePluginManifest(validManifest)).not.toThrow();
  });

  it("accepts controlled presentation metadata", () => {
    expect(() =>
      validatePluginManifest({
        ...validManifest,
        presentation: {
          icon: { iconId: "example-local", initials: "EL", label: "Example" },
          theme: {
            surface: "neutral",
            accentColor: "#64748B",
            foregroundMode: "dark",
          },
          supportsMultipleConnections: true,
        },
      }),
    ).not.toThrow();
  });

  it("accepts namespaced findingRules on the manifest", () => {
    expect(() =>
      validatePluginManifest({
        ...validManifest,
        capabilities: ["discover", "inspect", "evaluate_findings"],
        findingRules: [
          {
            id: "example-local.port-check",
            name: "Port check",
            description: "Ensures a local service port is present",
            category: "resource",
            defaultSeverity: "info",
          },
        ],
      }),
    ).not.toThrow();
  });

  it("rejects findingRules that are not namespaced to the plugin", () => {
    expect(() =>
      validatePluginManifest({
        ...validManifest,
        findingRules: [
          {
            id: "other.port-check",
            name: "Port check",
            description: "Bad namespace",
            category: "resource",
            defaultSeverity: "info",
          },
        ],
      }),
    ).toThrow(/namespaced/);
  });

  it("rejects free-form accent colors", () => {
    expect(() =>
      validatePluginManifest({
        ...validManifest,
        presentation: {
          theme: { surface: "brand", accentColor: "rgb(255,0,0)" },
        },
      }),
    ).toThrow(PluginValidationError);
  });

  it("validates discovered resources and plugin resources", () => {
    const discovered: DiscoveredResource = {
      providerResourceId: "svc-1",
      resourceType: "local.service",
      name: "API",
      metadata: { port: 3000 },
      schemaVersion: "1.0.0",
    };
    expect(() => validateDiscoveredResource(discovered)).not.toThrow();

    const resource: PluginResource = {
      id: "res-1",
      pluginId: "example-local",
      providerResourceId: "svc-1",
      resourceType: "local.service",
      name: "API",
      metadata: { port: 3000 },
      pluginVersion: "0.1.0",
      schemaVersion: "1.0.0",
    };
    expect(() => validatePluginResource(resource)).not.toThrow();

    expect(() =>
      validateDiscoveredResource({
        ...discovered,
        metadata: null as unknown as Record<string, unknown>,
      }),
    ).toThrow(PluginValidationError);
  });

  it("validates serializable change plans", () => {
    const plan: ChangePlan = {
      id: "plan-1",
      pluginId: "example-local",
      resourceId: "res-1",
      summary: "Update port",
      operations: [
        {
          id: "op-1",
          type: "update_attribute",
          description: "Set port to 4000",
          path: "attributes.port",
          before: 3000,
          after: 4000,
          requiresApproval: true,
        },
      ],
      warnings: [],
      destructive: false,
    };

    expect(() => validateChangePlan(plan)).not.toThrow();
    expect(JSON.parse(JSON.stringify(plan))).toEqual(plan);

    expect(() =>
      validateApprovedChangePlan({
        plan,
        approvalId: "approval-1",
        approvedAt: "1970-01-01T00:00:00.000Z",
        approvedOperationIds: ["op-1"],
      }),
    ).not.toThrow();

    expect(() =>
      validateApprovedChangePlan({
        plan,
        approvalId: "approval-1",
        approvedAt: "1970-01-01T00:00:00.000Z",
        approvedOperationIds: ["op-1", "op-1"],
      }),
    ).toThrow(PluginValidationError);
  });

  it("validates observed, apply, and verification results", () => {
    const observed: ObservedResourceState = {
      resourceId: "res-1",
      pluginId: "example-local",
      resourceType: "local.service",
      observedAt: "1970-01-01T00:00:00.000Z",
      status: "ready",
      attributes: { port: 4000 },
    };
    expect(() => validateObservedResourceState(observed)).not.toThrow();

    const applyResult: ApplyResult = {
      ok: true,
      message: "Applied",
      appliedOperationIds: ["op-1"],
      resultingState: observed,
    };
    expect(() => validateApplyResult(applyResult)).not.toThrow();

    const verification: VerificationResult = {
      ok: true,
      message: "Verified",
      observed,
      mismatches: [],
    };
    expect(() => validateVerificationResult(verification)).not.toThrow();
  });

  it("validates evaluate_findings context and results", () => {
    const context: EvaluateFindingsContext = {
      pluginId: "example-local",
      projectId: "project-1",
      connectionId: "conn-1",
      environments: [{ id: "env-1", name: "Local" }],
      resources: [{ resourceBindingId: "bind-1", resourceType: "local.service" }],
      observedStates: [
        {
          resourceBindingId: "bind-1",
          value: { access: "readable", value: "3000", sensitive: false },
        },
      ],
    };
    expect(() => validateEvaluateFindingsContext(context)).not.toThrow();

    const result: EvaluateFindingsResult = {
      detections: [validDetection()],
      warnings: [],
    };
    expect(() =>
      validateEvaluateFindingsResult(result, "example-local"),
    ).not.toThrow();
    expect(JSON.parse(JSON.stringify(result))).toEqual(result);
  });

  it("rejects unnamespaced rule ids and secret evidence", () => {
    expect(() =>
      validatePluginFindingDetection(
        validDetection({ ruleId: "unscoped.rule" }),
        "example-local",
      ),
    ).toThrow(/namespaced/);

    expect(() =>
      validatePluginFindingDetection(
        validDetection({
          evidence: [
            {
              type: "message",
              message: "token=super-secret-value",
            },
          ],
        }),
        "example-local",
      ),
    ).toThrow(/secret/);

    expect(() =>
      validatePluginFindingDetection(
        validDetection({
          fingerprintParts: [],
        }),
        "example-local",
      ),
    ).toThrow(/fingerprintParts/);

    expect(() =>
      validatePluginFindingDetection(
        validDetection({
          remediation: {
            type: "manual",
            label: "Fix it",
            instructions: "Rotate the key",
            run: () => undefined,
          } as unknown as PluginFindingDetection["remediation"],
        }),
        "example-local",
      ),
    ).toThrow(/executable|serializable|functions/);
  });
});
