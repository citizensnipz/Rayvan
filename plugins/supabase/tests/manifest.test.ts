import { describe, expect, it } from "vitest";
import { manifest } from "../src/manifest.js";

describe("@rayvan/plugin-supabase", () => {
  it("exports a manifest", () => {
    expect(manifest.id).toBe("supabase");
    expect(manifest.capabilities.length).toBeGreaterThan(0);
  });
});