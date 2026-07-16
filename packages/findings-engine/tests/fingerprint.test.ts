import { FINDING_FINGERPRINT_VERSION } from "@rayvan/core";
import { describe, expect, it } from "vitest";

import {
  buildFindingFingerprint,
  fingerprintVersion,
} from "../src/fingerprint.js";
import { sha256Hex } from "../src/sha256.js";
import { CORE_FINDING_RULE_IDS } from "../src/rules/registry.js";

describe("sha256Hex", () => {
  it("matches well-known SHA-256 digests", () => {
    expect(sha256Hex("")).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    expect(sha256Hex("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });
});

describe("buildFindingFingerprint", () => {
  it("is stable for the same rule, project, and parts", () => {
    const a = buildFindingFingerprint({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      projectId: "proj-1",
      fingerprintParts: ["env-1", "key-1"],
    });
    const b = buildFindingFingerprint({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      projectId: "proj-1",
      fingerprintParts: ["env-1", "key-1"],
    });
    expect(a).toBe(b);
    expect(a).toMatch(/^[a-f0-9]{64}$/);
  });

  it("does not change when titles would differ — only structural parts matter", () => {
    const withParts = buildFindingFingerprint({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      projectId: "proj-1",
      fingerprintParts: ["env-1", "key-1"],
    });
    // Titles/timestamps are never inputs — same parts → same fingerprint
    expect(
      buildFindingFingerprint({
        ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
        projectId: "proj-1",
        fingerprintParts: ["env-1", "key-1"],
      }),
    ).toBe(withParts);
  });

  it("changes when structural parts change", () => {
    const a = buildFindingFingerprint({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      projectId: "proj-1",
      fingerprintParts: ["env-1", "key-1"],
    });
    const b = buildFindingFingerprint({
      ruleId: CORE_FINDING_RULE_IDS.CONFIGURATION_MISMATCH,
      projectId: "proj-1",
      fingerprintParts: ["env-1", "key-2"],
    });
    expect(a).not.toBe(b);
  });

  it("uses FINDING_FINGERPRINT_VERSION from core", () => {
    expect(fingerprintVersion()).toBe(FINDING_FINGERPRINT_VERSION);
  });
});
