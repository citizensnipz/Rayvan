import { describe, expect, it } from "vitest";

import { createEnvironmentEvaluator } from "../src/evaluators/environment.js";
import { CORE_FINDING_RULE_IDS } from "../src/rules/registry.js";
import {
  ENV_DEV,
  NOW,
  PROJECT,
  emptyContext,
  makeEnvironment,
} from "./helpers.js";

const evaluator = createEnvironmentEvaluator();

describe("environment / resource evaluators", () => {
  it("emits unmapped for active discovered resource without environment binding", async () => {
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        discoveredResources: [
          {
            id: "res-1",
            pluginId: "plugin-a",
            connectionId: "conn-1",
            name: "Postgres",
            resourceType: "database",
            discoveryStatus: "active",
          },
        ],
        resourceBindings: [],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) => d.ruleId === CORE_FINDING_RULE_IDS.RESOURCE_UNMAPPED,
      ),
    ).toBe(true);
  });

  it("emits missing when active binding points at inaccessible resource", async () => {
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        discoveredResources: [
          {
            id: "res-1",
            pluginId: "plugin-a",
            connectionId: "conn-1",
            name: "Gone",
            resourceType: "service",
            discoveryStatus: "inaccessible",
          },
        ],
        resourceBindings: [
          {
            id: "binding-1",
            projectId: PROJECT,
            environmentId: ENV_DEV,
            discoveredResourceId: "res-1",
            pluginId: "plugin-a",
            connectionId: "conn-1",
            bindingStatus: "active",
          },
        ],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) => d.ruleId === CORE_FINDING_RULE_IDS.RESOURCE_MISSING,
      ),
    ).toBe(true);
  });

  it("does not emit no-resources for local_only environments", async () => {
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        environments: [
          makeEnvironment(ENV_DEV, {
            status: "local_only",
            createdAt: "2025-01-01T00:00:00.000Z",
          }),
        ],
        resourceBindings: [],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) => d.ruleId === CORE_FINDING_RULE_IDS.ENVIRONMENT_NO_RESOURCES,
      ),
    ).toBe(false);
  });

  it("skips no-resources for newly created environments (<24h)", async () => {
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        environments: [
          makeEnvironment(ENV_DEV, {
            status: "healthy",
            createdAt: "2026-07-16T06:00:00.000Z",
          }),
        ],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) => d.ruleId === CORE_FINDING_RULE_IDS.ENVIRONMENT_NO_RESOURCES,
      ),
    ).toBe(false);
  });

  it("emits pending mapping suggestion per suggestion id", async () => {
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        mappingSuggestions: [
          {
            id: "sug-1",
            projectId: PROJECT,
            connectionId: "conn-1",
            discoveredResourceId: "res-1",
            suggestedEnvironmentId: ENV_DEV,
            suggestedEnvironmentName: "Development",
            status: "pending",
          },
        ],
      }),
      now: NOW,
    });
    const pending = detections.filter(
      (d) =>
        d.ruleId ===
        CORE_FINDING_RULE_IDS.ENVIRONMENT_PENDING_MAPPING_SUGGESTION,
    );
    expect(pending).toHaveLength(1);
    expect(pending[0]!.fingerprintParts).toContain("sug-1");
  });
});
