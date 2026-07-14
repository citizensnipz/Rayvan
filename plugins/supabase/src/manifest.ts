import {
  RAYVAN_PLUGIN_API_VERSION,
  type PluginManifest,
} from "@rayvan/plugin-sdk";

export const manifest: PluginManifest = {
  id: "supabase",
  name: "Supabase",
  description: "Supabase provider plugin (placeholder)",
  version: "0.0.1",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [],
  permissions: [],
  resourceTypes: [],
};
