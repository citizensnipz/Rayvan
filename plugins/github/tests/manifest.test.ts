import { describe, expect, it } from "vitest";
import { InProcessPluginRegistry } from "@rayvan/plugin-sdk";
import { manifest, plugin } from "../src/index.js";

describe("@rayvan/plugin-github", () => {
  it("exports a placeholder manifest that can be registered", () => {
    expect(manifest.id).toBe("github");
    expect(manifest.capabilities).toEqual([]);

    const registry = new InProcessPluginRegistry();
    registry.register(plugin);
    expect(registry.get("github")?.manifest.name).toBe("GitHub");
  });
});
