import { describe, expect, it } from "vitest";

import {
  CORE_FINDING_RULE_IDS,
  listCoreFindingRuleIds,
  listCoreFindingRules,
} from "../src/rules/registry.js";

const EXPECTED_IDS = [
  "rayvan.configuration.missing-required",
  "rayvan.configuration.mismatch",
  "rayvan.configuration.unapplied",
  "rayvan.configuration.remote-changed",
  "rayvan.configuration.partially-applied",
  "rayvan.configuration.comparison-unavailable",
  "rayvan.configuration.unmanaged",
  "rayvan.configuration.inspection-stale",
  "rayvan.resource.unmapped",
  "rayvan.resource.missing",
  "rayvan.environment.no-resources",
  "rayvan.environment.pending-mapping-suggestion",
  "rayvan.integration.connection-expired",
  "rayvan.integration.connection-revoked",
  "rayvan.integration.authentication-failed",
  "rayvan.integration.permission-missing",
  "rayvan.integration.plugin-disabled",
  "rayvan.integration.plugin-incompatible",
  "rayvan.integration.sync-failure",
  "rayvan.integration.credential-missing",
  "rayvan.change.apply-failed",
  "rayvan.change.apply-partial",
  "rayvan.change.verification-failed",
  "rayvan.change.plan-stale",
] as const;

describe("core finding rules", () => {
  it("exposes stable rule IDs", () => {
    expect(listCoreFindingRuleIds()).toEqual([...EXPECTED_IDS]);
    expect(Object.values(CORE_FINDING_RULE_IDS).sort()).toEqual(
      [...EXPECTED_IDS].sort(),
    );
  });

  it("defines a rule for every stable ID", () => {
    const ids = new Set(listCoreFindingRules().map((rule) => rule.id));
    for (const id of EXPECTED_IDS) {
      expect(ids.has(id)).toBe(true);
    }
  });
});
