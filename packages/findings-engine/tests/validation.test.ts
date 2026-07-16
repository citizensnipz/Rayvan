import { describe, expect, it } from "vitest";

import {
  FindingValidationError,
  validateDetection,
} from "../src/validation.js";
import { PROJECT } from "./helpers.js";

function baseDetection(
  evidence: Parameters<typeof validateDetection>[0]["evidence"],
) {
  return {
    ruleId: "rayvan.test.rule",
    projectId: PROJECT,
    title: "Test",
    summary: "Summary",
    scope: {},
    evidence,
    fingerprintParts: ["rayvan.test.rule", "x"],
  };
}

describe("validateDetection secrets", () => {
  it("rejects message evidence that looks like a bearer token", () => {
    expect(() =>
      validateDetection(
        baseDetection([
          {
            type: "message",
            message: "Auth failed with Bearer sk-live_abcdefghijklmnopqrst",
          },
        ]),
      ),
    ).toThrow(FindingValidationError);
  });

  it("rejects connection_error safeMessage with api_key assignment", () => {
    expect(() =>
      validateDetection(
        baseDetection([
          {
            type: "connection_error",
            connectionId: "conn-1",
            safeMessage: "api_key=super-secret-value-here",
          },
        ]),
      ),
    ).toThrow(/secret/);
  });

  it("rejects readable SafeFindingValue containing obvious secrets", () => {
    expect(() =>
      validateDetection(
        baseDetection([
          {
            type: "configuration_comparison",
            configurationKeyId: "key-1",
            environmentId: "env-1",
            expectedState: {
              access: "readable",
              value: "ghp_abcdefghijklmnopqrstuvwxyz12",
              sensitive: false,
            },
            observedStates: [],
          },
        ]),
      ),
    ).toThrow(/secret/);
  });

  it("accepts fingerprint SafeFindingValue shapes", () => {
    expect(() =>
      validateDetection(
        baseDetection([
          {
            type: "configuration_comparison",
            configurationKeyId: "key-1",
            environmentId: "env-1",
            expectedState: {
              access: "fingerprint",
              fingerprint: "fp-abc",
              sensitive: true,
            },
            observedStates: [
              {
                value: {
                  access: "masked",
                  maskedValue: "••••last4",
                  sensitive: true,
                },
              },
            ],
          },
        ]),
      ),
    ).not.toThrow();
  });

  it("rejects readable values marked sensitive", () => {
    expect(() =>
      validateDetection(
        baseDetection([
          {
            type: "configuration_comparison",
            configurationKeyId: "key-1",
            environmentId: "env-1",
            expectedState: {
              access: "readable",
              value: "ok",
              // @ts-expect-error intentional invalid shape
              sensitive: true,
            },
            observedStates: [],
          },
        ]),
      ),
    ).toThrow(/sensitive must be false/);
  });
});
