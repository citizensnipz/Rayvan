import { describe, expect, it } from "vitest";
import { isNonEmptyString } from "../src/validation/index.js";

describe("@rayvan/shared", () => {
  it("validates non-empty strings", () => {
    expect(isNonEmptyString("rayvan")).toBe(true);
    expect(isNonEmptyString("  ")).toBe(false);
  });
});
