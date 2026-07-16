import type { FindingDetection } from "@rayvan/core";
import { FINDING_SCHEMA_VERSION, findingId } from "@rayvan/core";
import { describe, expect, it } from "vitest";

import { FindingEngine } from "../src/engine/finding-engine.js";
import type { FindingEvaluator } from "../src/evaluators/types.js";
import { buildFindingFingerprint } from "../src/fingerprint.js";
import { CORE_FINDING_RULE_IDS } from "../src/rules/registry.js";
import {
  NOW,
  PROJECT,
  createTestEngine,
  createTestRepos,
  emptyContext,
  makeFindingRecord,
  makeKey,
  makeDesired,
} from "./helpers.js";

describe("FindingEngine", () => {
  it("acknowledge does not resolve on subsequent evaluation", async () => {
    const { engine, repos } = createTestEngine();
    const context = emptyContext({
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
    });

    const first = await engine.evaluateProject(PROJECT, {
      trigger: "manual",
      now: NOW,
      context,
    });
    expect(first.created.length).toBeGreaterThan(0);
    const created = first.created.find(
      (r) =>
        r.ruleId === CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
    )!;
    expect(created).toBeDefined();

    await repos.findings.save({
      ...created,
      status: "acknowledged",
      acknowledgedAt: NOW,
      acknowledgedBy: { kind: "user", id: "u1" },
    });

    const second = await engine.evaluateProject(PROJECT, {
      trigger: "manual",
      now: "2026-07-16T13:00:00.000Z",
      context,
    });
    const still = await repos.findings.getById(created.id);
    expect(still?.status).toBe("acknowledged");
    expect(second.resolved.find((r) => r.id === created.id)).toBeUndefined();
    expect(still!.occurrenceCount).toBeGreaterThan(created.occurrenceCount);
  });

  it("preserves previous plugin findings when that plugin evaluator fails", async () => {
    const repos = createTestRepos();
    const pluginRuleId = "plugin.example.issue";
    const fingerprint = buildFindingFingerprint({
      ruleId: pluginRuleId,
      projectId: PROJECT,
      fingerprintParts: [pluginRuleId, "conn-1"],
    });
    const existing = makeFindingRecord({
      id: findingId("finding-plugin-1"),
      ruleId: pluginRuleId,
      fingerprint,
      source: { type: "plugin", pluginId: "example" },
      category: "other",
      status: "open",
      connectionId: "conn-1",
      title: "Plugin issue",
      summary: "Still open",
      schemaVersion: FINDING_SCHEMA_VERSION,
    });
    repos.findings.seed(existing);

    const failingPlugin: FindingEvaluator = {
      id: "plugin.example",
      pluginId: "example",
      ruleIds: [pluginRuleId],
      evaluate: async () => {
        throw new Error("plugin boom");
      },
    };

    const engine = new FindingEngine({
      repositories: repos,
      extraRules: [
        {
          id: pluginRuleId,
          name: "Plugin issue",
          description: "test",
          source: { type: "plugin", pluginId: "example" },
          category: "other",
          defaultSeverity: "warning",
          enabledByDefault: true,
          supportedObjectTypes: ["connection"],
        },
      ],
    });

    const result = await engine.evaluateProject(PROJECT, {
      trigger: "manual",
      now: NOW,
      context: emptyContext(),
      pluginEvaluators: [failingPlugin],
    });

    expect(result.run.status).toBe("partially_succeeded");
    expect(result.errors.some((e) => e.pluginId === "example")).toBe(true);
    const preserved = await repos.findings.getById(existing.id);
    expect(preserved?.status).toBe("open");
    expect(result.resolved.find((r) => r.id === existing.id)).toBeUndefined();
  });

  it("serializes concurrent evaluations for the same project (mutex)", async () => {
    const context = emptyContext({
      keys: [makeKey({ name: "X", required: true })],
      desired: [makeDesired("key-X")],
    });

    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let active = 0;
    let maxActive = 0;

    const slowEvaluator: FindingEvaluator = {
      id: "slow",
      ruleIds: [CORE_FINDING_RULE_IDS.CONFIGURATION_MISSING_REQUIRED],
      evaluate: async () => {
        active += 1;
        maxActive = Math.max(maxActive, active);
        await gate;
        active -= 1;
        return {
          detections: [] as FindingDetection[],
          evaluatedRuleIds: [CORE_FINDING_RULE_IDS.CONFIGURATION_MISSING_REQUIRED],
        };
      },
    };

    const engineWithSlow = new FindingEngine({
      repositories: createTestRepos(),
      coreEvaluators: [slowEvaluator],
    });

    const first = engineWithSlow.evaluateProject(PROJECT, {
      trigger: "manual",
      now: NOW,
      context,
    });
    const second = engineWithSlow.evaluateEnvironment(PROJECT, "env-dev", {
      trigger: "manual",
      now: NOW,
      context,
    });

    // Allow both to reach the lock; only one evaluator should run at a time.
    await Promise.resolve();
    expect(maxActive).toBeLessThanOrEqual(1);
    release();
    await Promise.all([first, second]);
    expect(maxActive).toBe(1);
  });

  it("resolves stale open findings when rule scope succeeds without detection", async () => {
    const { engine, repos } = createTestEngine();
    const fingerprint = buildFindingFingerprint({
      ruleId: CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
      projectId: PROJECT,
      fingerprintParts: [
        CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
        "conn-1",
      ],
    });
    const stale = makeFindingRecord({
      ruleId: CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
      fingerprint,
      connectionId: "conn-1",
      status: "open",
      category: "integration",
    });
    repos.findings.seed(stale);

    const result = await engine.evaluateProject(PROJECT, {
      trigger: "manual",
      now: NOW,
      context: emptyContext({
        connections: [
          {
            id: "conn-1",
            pluginId: "vercel",
            name: "Vercel",
            status: "connected",
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
    });

    expect(result.resolved.some((r) => r.id === stale.id)).toBe(true);
    const after = await repos.findings.getById(stale.id);
    expect(after?.status).toBe("resolved");
  });

  it("does not resolve open findings when evaluation is cancelled", async () => {
    const repos = createTestRepos();
    const fingerprint = buildFindingFingerprint({
      ruleId: CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
      projectId: PROJECT,
      fingerprintParts: [
        CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
        "conn-1",
      ],
    });
    const open = makeFindingRecord({
      ruleId: CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
      fingerprint,
      connectionId: "conn-1",
      status: "open",
      category: "integration",
    });
    repos.findings.seed(open);

    const controller = new AbortController();
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    const blocking: FindingEvaluator = {
      id: "blocking",
      ruleIds: [CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED],
      evaluate: async () => {
        controller.abort();
        await gate;
        return {
          detections: [],
          evaluatedRuleIds: [
            CORE_FINDING_RULE_IDS.INTEGRATION_CONNECTION_EXPIRED,
          ],
        };
      },
    };

    const engineWithBlock = new FindingEngine({
      repositories: repos,
      coreEvaluators: [blocking],
    });

    const resultPromise = engineWithBlock.evaluateProject(PROJECT, {
      trigger: "manual",
      now: NOW,
      context: emptyContext(),
      abortSignal: controller.signal,
    });
    await Promise.resolve();
    release();
    const result = await resultPromise;

    expect(result.run.status).toBe("cancelled");
    expect(result.resolved).toHaveLength(0);
    expect(result.created).toHaveLength(0);
    const preserved = await repos.findings.getById(open.id);
    expect(preserved?.status).toBe("open");
  });

  it("does not resolve plan-stale findings when changePlans data is unavailable", async () => {
    const { engine, repos } = createTestEngine();
    const fingerprint = buildFindingFingerprint({
      ruleId: CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE,
      projectId: PROJECT,
      fingerprintParts: [CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE, "plan-1"],
    });
    const stale = makeFindingRecord({
      ruleId: CORE_FINDING_RULE_IDS.CHANGE_PLAN_STALE,
      fingerprint,
      changePlanId: "plan-1",
      status: "open",
      category: "change",
    });
    repos.findings.seed(stale);

    const result = await engine.evaluateProject(PROJECT, {
      trigger: "manual",
      now: NOW,
      // changePlans omitted (undefined) — rule must not count as evaluated
      context: emptyContext({
        changeApplies: [],
        changeVerifications: [],
      }),
    });

    expect(result.resolved.find((r) => r.id === stale.id)).toBeUndefined();
    const after = await repos.findings.getById(stale.id);
    expect(after?.status).toBe("open");
  });
});
