import { describe, expect, it } from "vitest";

import { createIntegrationEvaluator } from "../src/evaluators/integration.js";
import { CORE_FINDING_RULE_IDS } from "../src/rules/registry.js";
import { NOW, PROJECT, emptyContext } from "./helpers.js";

const evaluator = createIntegrationEvaluator();

describe("integration evaluators", () => {
  it("emits connection-expired", async () => {
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        connections: [
          {
            id: "conn-1",
            pluginId: "vercel",
            name: "Vercel",
            status: "expired",
            credentialReferenceId: "cred-1",
          },
        ],
        installedPlugins: [
          {
            id: "inst-1",
            pluginId: "vercel",
            pluginVersion: "1.0.0",
            status: "installed",
            enabled: true,
          },
        ],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) =>
          d.ruleId === CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
      ),
    ).toBe(true);
  });

  it("emits credential-missing when credentialReferenceId absent", async () => {
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        connections: [
          {
            id: "conn-1",
            pluginId: "vercel",
            name: "Vercel",
            status: "connected",
          },
        ],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) =>
          d.ruleId === CORE_FINDING_RULE_IDS.INTEGRATION_CREDENTIAL_MISSING,
      ),
    ).toBe(true);
  });

  it("emits plugin-disabled and sync-failure", async () => {
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        connections: [
          {
            id: "conn-1",
            pluginId: "railway",
            name: "Railway",
            status: "connected",
            credentialReferenceId: "cred-1",
            lastFailedSyncAt: "2026-07-16T11:00:00.000Z",
            lastSuccessfulSyncAt: "2026-07-15T11:00:00.000Z",
          },
        ],
        installedPlugins: [
          {
            id: "inst-1",
            pluginId: "railway",
            pluginVersion: "1.0.0",
            status: "disabled",
            enabled: false,
          },
        ],
      }),
      now: NOW,
    });
    const ids = new Set(detections.map((d) => d.ruleId));
    expect(ids.has(CORE_FINDING_RULE_IDS.INTEGRATION_PLUGIN_DISABLED)).toBe(
      true,
    );
    expect(ids.has(CORE_FINDING_RULE_IDS.INTEGRATION_SYNC_FAILURE)).toBe(true);
  });

  it("scopes to a single connection", async () => {
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: {
        type: "connection",
        projectId: PROJECT,
        connectionId: "conn-2",
      },
      context: emptyContext({
        connections: [
          {
            id: "conn-1",
            pluginId: "a",
            name: "A",
            status: "expired",
            credentialReferenceId: "c1",
          },
          {
            id: "conn-2",
            pluginId: "b",
            name: "B",
            status: "revoked",
            credentialReferenceId: "c2",
          },
        ],
      }),
      now: NOW,
    });
    expect(
      detections.every((d) => d.scope.connectionId === "conn-2"),
    ).toBe(true);
    expect(
      detections.some(
        (d) =>
          d.ruleId === CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_REVOKED,
      ),
    ).toBe(true);
  });
});
