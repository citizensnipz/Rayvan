import { describe, expect, it } from "vitest";
import { manifest } from "../src/manifest.js";

describe("@rayvan/plugin-github", () => {
  it("exports a manifest", () => {
    expect(manifest.id).toBe("github");
    expect(manifest.capabilities.length).toBeGreaterThan(0);
  });
});
