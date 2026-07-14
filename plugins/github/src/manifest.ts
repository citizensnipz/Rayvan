import {
  RAYVAN_PLUGIN_API_VERSION,
  type PluginManifest,
} from "@rayvan/plugin-sdk";

export const manifest: PluginManifest = {
  id: "github",
  name: "GitHub",
  description: "Connect Rayvan to GitHub repositories and pull requests.",
  version: "0.0.1",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [],
  permissions: [],
  resourceTypes: [],
  presentation: {
    icon: { iconId: "github", initials: "GH", label: "GitHub" },
    theme: {
      surface: "dark",
      accentColor: "#F0F6FC",
      foregroundMode: "light",
    },
    supportsMultipleConnections: true,
  },
};
