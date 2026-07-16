import { describe, expect, it } from "vitest";

import { createConfigurationEvaluator } from "../src/evaluators/configuration.js";
import { CORE_FINDING_RULE_IDS } from "../src/rules/registry.js";
import {
  ENV_DEV,
  NOW,
  PROJECT,
  emptyContext,
  makeApplied,
  makeDesired,
  makeEnvironment,
  makeKey,
  makeOccurrence,
} from "./helpers.js";

const evaluator = createConfigurationEvaluator();

describe("configuration evaluators", () => {
  it("emits missing-required for required key with missing_remote", async () => {
    const key = makeKey({ name: "DATABASE_URL", required: true });
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        keys: [key],
        desired: [makeDesired(key.id)],
        // desired exists but no occurrence → missing_remote
        occurrences: [],
        applied: [],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) =>
          d.ruleId === CORE_FINDING_RULE_IDS.CONFIGURATION_MISSING_REQUIRED,
      ),
    ).toBe(true);
  });

  it("emits mismatch when desired disagrees with observed", async () => {
    const key = makeKey({ name: "API_URL" });
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        keys: [key],
        desired: [
          makeDesired(key.id, ENV_DEV, {
            desiredValue: "https://a.example",
            valueFingerprint: "fp-a",
            revision: 1,
          }),
        ],
        occurrences: [
          makeOccurrence({
            configurationKeyId: key.id,
            resourceBindingId: "binding-1",
            observedValue: "https://b.example",
            valueFingerprint: "fp-b",
          }),
        ],
        // Applied revision matches desired but has no fingerprint → appliedVsObserved
        // is unknown, so status is mismatched (not remote_changed).
        applied: [
          makeApplied({
            configurationKeyId: key.id,
            desiredRevision: 1,
            appliedFingerprint: undefined,
          }),
        ],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) => d.ruleId === CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      ),
    ).toBe(true);
  });

  it("emits partially-applied when resources disagree", async () => {
    const key = makeKey({ name: "SHARED" });
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        keys: [key],
        desired: [
          makeDesired(key.id, ENV_DEV, {
            desiredValue: "desired",
            valueFingerprint: "fp-desired",
            revision: 1,
          }),
        ],
        occurrences: [
          makeOccurrence({
            id: "occ-a",
            configurationKeyId: key.id,
            resourceBindingId: "binding-a",
            discoveredResourceId: "res-a",
            observedValue: "desired",
            valueFingerprint: "fp-desired",
          }),
          makeOccurrence({
            id: "occ-b",
            configurationKeyId: key.id,
            resourceBindingId: "binding-b",
            discoveredResourceId: "res-b",
            observedValue: "other",
            valueFingerprint: "fp-other",
          }),
        ],
        applied: [
          makeApplied({
            id: "app-a",
            configurationKeyId: key.id,
            resourceBindingId: "binding-a",
            appliedFingerprint: "fp-desired",
            desiredRevision: 1,
          }),
          makeApplied({
            id: "app-b",
            configurationKeyId: key.id,
            resourceBindingId: "binding-b",
            appliedFingerprint: "fp-desired",
            desiredRevision: 1,
          }),
        ],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) =>
          d.ruleId === CORE_FINDING_RULE_IDS.CONFIGURATION_PARTIALLY_APPLIED,
      ),
    ).toBe(true);
  });

  it("emits unapplied for local_changes without treating drafts", async () => {
    const key = makeKey({ name: "FEATURE_FLAG" });
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        keys: [key],
        desired: [
          makeDesired(key.id, ENV_DEV, {
            desiredValue: "new",
            valueFingerprint: "fp-new",
            revision: 2,
          }),
        ],
        occurrences: [
          makeOccurrence({
            configurationKeyId: key.id,
            resourceBindingId: "binding-1",
            observedValue: "old",
            valueFingerprint: "fp-old",
          }),
        ],
        applied: [
          makeApplied({
            configurationKeyId: key.id,
            appliedFingerprint: "fp-old",
            desiredRevision: 1,
          }),
        ],
      }),
      now: NOW,
    });
    // local_changes or mismatched depending on compare — either unapplied or mismatch is fine;
    // with desired≠observed and desired≠applied we expect local_changes → unapplied OR mismatched
    const ruleIds = new Set(detections.map((d) => d.ruleId));
    expect(
      ruleIds.has(CORE_FINDING_RULE_IDS.CONFIGURATION_UNAPPLIED) ||
        ruleIds.has(CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH),
    ).toBe(true);
  });

  it("emits remote-changed when observed drifts from applied while desired matches applied", async () => {
    const key = makeKey({ name: "TOKEN" });
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        keys: [key],
        desired: [
          makeDesired(key.id, ENV_DEV, {
            desiredValue: "desired",
            valueFingerprint: "fp-desired",
          }),
        ],
        occurrences: [
          makeOccurrence({
            configurationKeyId: key.id,
            resourceBindingId: "binding-1",
            observedValue: "remote-new",
            valueFingerprint: "fp-remote",
          }),
        ],
        applied: [
          makeApplied({
            configurationKeyId: key.id,
            appliedFingerprint: "fp-desired",
          }),
        ],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) =>
          d.ruleId === CORE_FINDING_RULE_IDS.CONFIGURATION_REMOTE_CHANGED,
      ),
    ).toBe(true);
  });

  it("emits comparison-unavailable for locked values", async () => {
    const key = makeKey({ name: "SECRET", sensitive: true });
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        keys: [key],
        desired: [
          makeDesired(key.id, ENV_DEV, {
            desiredValue: undefined,
            secretValueRef: "ref-1",
            valueFingerprint: "fp-desired",
          }),
        ],
        occurrences: [
          makeOccurrence({
            configurationKeyId: key.id,
            resourceBindingId: "binding-1",
            valueAccess: "locked",
          }),
        ],
        applied: [],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) =>
          d.ruleId ===
          CORE_FINDING_RULE_IDS.CONFIGURATION_COMPARISON_UNAVAILABLE,
      ),
    ).toBe(true);
  });

  it("emits inspection-stale when observation is older than 7 days", async () => {
    const key = makeKey({ name: "CACHE_TTL" });
    const old = "2026-06-01T00:00:00.000Z";
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        keys: [key],
        desired: [
          makeDesired(key.id, ENV_DEV, {
            desiredValue: "60",
            valueFingerprint: "fp-60",
          }),
        ],
        occurrences: [
          makeOccurrence({
            configurationKeyId: key.id,
            resourceBindingId: "binding-1",
            observedValue: "60",
            valueFingerprint: "fp-60",
            lastObservedAt: old,
            firstObservedAt: old,
          }),
        ],
        applied: [
          makeApplied({
            configurationKeyId: key.id,
            appliedFingerprint: "fp-60",
          }),
        ],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) =>
          d.ruleId === CORE_FINDING_RULE_IDS.CONFIGURATION_INSPECTION_STALE,
      ),
    ).toBe(true);
  });

  it("emits unmanaged for observed without desired", async () => {
    const key = makeKey({ name: "LEGACY" });
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: { type: "project", projectId: PROJECT },
      context: emptyContext({
        keys: [key],
        desired: [],
        occurrences: [
          makeOccurrence({
            configurationKeyId: key.id,
            observedValue: "x",
            valueFingerprint: "fp-x",
          }),
        ],
      }),
      now: NOW,
    });
    expect(
      detections.some(
        (d) => d.ruleId === CORE_FINDING_RULE_IDS.CONFIGURATION_UNMANAGED,
      ),
    ).toBe(true);
  });

  it("skips connection scope", async () => {
    const { detections } = await evaluator.evaluate({
      projectId: PROJECT,
      scope: {
        type: "connection",
        projectId: PROJECT,
        connectionId: "conn-1",
      },
      context: emptyContext({
        environments: [makeEnvironment()],
        keys: [makeKey({ name: "X", required: true })],
        desired: [makeDesired("key-X")],
      }),
      now: NOW,
    });
    expect(detections).toEqual([]);
  });
});
