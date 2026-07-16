import { describe, expect, it } from "vitest";

import {
  acknowledgeFinding,
  dismissFinding,
  reopenFinding,
  resolveFinding,
  suppressFinding,
} from "../src/engine/lifecycle.js";
import { makeFindingRecord, NOW } from "./helpers.js";
import { CORE_FINDING_RULE_IDS } from "../src/rules/registry.js";

const USER = { kind: "user" as const, id: "user-1", displayName: "Ada" };

describe("lifecycle helpers", () => {
  it("acknowledge moves open → acknowledged and does not resolve", () => {
    const record = makeFindingRecord({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      fingerprint: "fp-1",
      status: "open",
    });
    const result = acknowledgeFinding(record, USER, NOW);
    expect(result.record.status).toBe("acknowledged");
    expect(result.record.acknowledgedBy).toEqual(USER);
    expect(result.record.resolvedAt).toBeUndefined();
    expect(result.event.type).toBe("acknowledged");
  });

  it("dismiss sets dismissed with optional reason", () => {
    const record = makeFindingRecord({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      fingerprint: "fp-1",
    });
    const result = dismissFinding(record, USER, NOW, "not relevant");
    expect(result.record.status).toBe("dismissed");
    expect(result.record.dismissalReason).toBe("not relevant");
  });

  it("dismiss rejects resolved findings", () => {
    const record = makeFindingRecord({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      fingerprint: "fp-1",
      status: "resolved",
      resolvedAt: NOW,
      resolution: { source: "automatic", resolvedBy: USER },
    });
    expect(() => dismissFinding(record, USER, NOW)).toThrow(/resolved/);
  });

  it("suppress sets suppressedUntil and status", () => {
    const until = "2026-08-01T00:00:00.000Z";
    const record = makeFindingRecord({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      fingerprint: "fp-1",
    });
    const result = suppressFinding(record, USER, NOW, until);
    expect(result.record.status).toBe("suppressed");
    expect(result.record.suppressedUntil).toBe(until);
  });

  it("suppress rejects resolved findings", () => {
    const record = makeFindingRecord({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      fingerprint: "fp-1",
      status: "resolved",
      resolvedAt: NOW,
      resolution: { source: "manual", resolvedBy: USER },
    });
    expect(() =>
      suppressFinding(record, USER, NOW, "2026-08-01T00:00:00.000Z"),
    ).toThrow(/resolved/);
  });

  it("resolve sets automatic resolution", () => {
    const record = makeFindingRecord({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      fingerprint: "fp-1",
    });
    const result = resolveFinding(record, USER, NOW, "automatic", "cleared");
    expect(result.record.status).toBe("resolved");
    expect(result.record.resolution?.source).toBe("automatic");
  });

  it("reopen clears resolve/dismiss/suppress and acknowledgement fields", () => {
    const record = makeFindingRecord({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      fingerprint: "fp-1",
      status: "resolved",
      resolvedAt: NOW,
      resolution: { source: "automatic", resolvedBy: USER },
      acknowledgedAt: NOW,
      acknowledgedBy: USER,
      dismissedAt: NOW,
      dismissedBy: USER,
      dismissalReason: "old",
      suppressedUntil: "2026-08-01T00:00:00.000Z",
    });
    const result = reopenFinding(
      record,
      {
        ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
        projectId: record.projectId,
        title: "Again",
        summary: "Reappeared",
        scope: {},
        evidence: [],
        fingerprintParts: [CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH, "x"],
      },
      USER,
      "2026-07-16T14:00:00.000Z",
    );
    expect(result.record.status).toBe("open");
    expect(result.record.resolvedAt).toBeUndefined();
    expect(result.record.resolution).toBeUndefined();
    expect(result.record.dismissedAt).toBeUndefined();
    expect(result.record.dismissedBy).toBeUndefined();
    expect(result.record.dismissalReason).toBeUndefined();
    expect(result.record.suppressedUntil).toBeUndefined();
    expect(result.record.acknowledgedAt).toBeUndefined();
    expect(result.record.acknowledgedBy).toBeUndefined();
  });
});
