import {
  RAYVAN_PLUGIN_API_VERSION,
  type PluginManifest,
} from "@rayvan/plugin-sdk";

export const manifest: PluginManifest = {
  id: "github",
  name: "GitHub",
  description: "GitHub provider plugin (placeholder)",
  version: "0.0.1",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [],
  permissions: [],
  resourceTypes: [],
};
