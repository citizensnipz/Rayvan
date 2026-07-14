import { describe, expect, it } from "vitest";
import { manifest } from "../src/manifest.js";

describe("@rayvan/plugin-vercel", () => {
  it("exports a manifest", () => {
    expect(manifest.id).toBe("vercel");
    expect(manifest.capabilities.length).toBeGreaterThan(0);
  });
});