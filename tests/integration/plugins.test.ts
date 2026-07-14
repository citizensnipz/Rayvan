import { describe, expect, it } from "vitest";
import {
  InProcessPluginRegistry,
  RAYVAN_PLUGIN_API_VERSION,
} from "../../packages/plugin-sdk/src/index.ts";
import { plugin as exampleLocalPlugin } from "../../plugins/example-local/src/index.ts";
import { plugin as githubPlugin } from "../../plugins/github/src/index.ts";
import { plugin as runpodPlugin } from "../../plugins/runpod/src/index.ts";
import { plugin as supabasePlugin } from "../../plugins/supabase/src/index.ts";
import { plugin as vercelPlugin } from "../../plugins/vercel/src/index.ts";

const plugins = [
  exampleLocalPlugin,
  githubPlugin,
  vercelPlugin,
  supabasePlugin,
  runpodPlugin,
];

describe("bundled plugins", () => {
  it("exports unique manifests on the current API version", () => {
    const ids = plugins.map((plugin) => plugin.manifest.id);
    expect(new Set(ids).size).toBe(ids.length);

    for (const plugin of plugins) {
      expect(plugin.manifest.rayvanApiVersion).toBe(RAYVAN_PLUGIN_API_VERSION);
      expect(plugin.manifest.publisher).toBe("rayvan");
    }
  });

  it("registers all bundled plugins in the in-process registry", () => {
    const registry = new InProcessPluginRegistry();
    for (const plugin of plugins) {
      registry.register(plugin);
    }

    expect(registry.list()).toHaveLength(plugins.length);
    expect(registry.supports("example-local", "discover")).toBe(true);
    expect(registry.supports("vercel", "discover")).toBe(false);
  });
});
