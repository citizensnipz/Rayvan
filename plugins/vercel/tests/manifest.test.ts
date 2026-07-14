import { describe, expect, it } from "vitest";
import { InProcessPluginRegistry } from "@rayvan/plugin-sdk";
import { manifest, plugin } from "../src/index.js";

describe("@rayvan/plugin-vercel", () => {
  it("exports a placeholder manifest that can be registered", () => {
    expect(manifest.id).toBe("vercel");
    expect(manifest.capabilities).toEqual([]);
    expect(manifest.permissions).toEqual([]);

    const registry = new InProcessPluginRegistry();
    registry.register(plugin);
    expect(registry.get("vercel")?.manifest.name).toBe("Vercel");
  });
});
