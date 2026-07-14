import { describe, expect, it } from "vitest";
import { InProcessPluginRegistry } from "@rayvan/plugin-sdk";
import { manifest, plugin } from "../src/index.js";

describe("@rayvan/plugin-runpod", () => {
  it("exports a placeholder manifest that can be registered", () => {
    expect(manifest.id).toBe("runpod");
    expect(manifest.capabilities).toEqual([]);

    const registry = new InProcessPluginRegistry();
    registry.register(plugin);
    expect(registry.get("runpod")?.manifest.name).toBe("RunPod");
  });
});
