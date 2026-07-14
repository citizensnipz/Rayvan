import { describe, expect, it } from "vitest";
import { manifest } from "../src/manifest.js";

describe("@rayvan/plugin-runpod", () => {
  it("exports a manifest", () => {
    expect(manifest.id).toBe("runpod");
    expect(manifest.capabilities.length).toBeGreaterThan(0);
  });
});