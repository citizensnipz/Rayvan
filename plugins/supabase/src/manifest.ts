import {
  RAYVAN_PLUGIN_API_VERSION,
  type PluginManifest,
} from "@rayvan/plugin-sdk";

export const manifest: PluginManifest = {
  id: "supabase",
  name: "Supabase",
  description: "Connect Rayvan to Supabase projects and databases.",
  version: "0.0.1",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [],
  permissions: [],
  resourceTypes: [],
  presentation: {
    icon: { iconId: "supabase", initials: "S", label: "Supabase" },
    theme: {
      surface: "dark",
      accentColor: "#3ECF8E",
      foregroundMode: "light",
    },
    supportsMultipleConnections: true,
  },
};
