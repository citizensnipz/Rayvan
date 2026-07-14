import {
  RAYVAN_PLUGIN_API_VERSION,
  type PluginManifest,
} from "@rayvan/plugin-sdk";

export const manifest: PluginManifest = {
  id: "vercel",
  name: "Vercel",
  description: "Connect Rayvan to Vercel projects and deployments.",
  version: "0.0.1",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [],
  permissions: [],
  resourceTypes: [],
  presentation: {
    icon: { iconId: "vercel", initials: "V", label: "Vercel" },
    theme: {
      surface: "dark",
      accentColor: "#FFFFFF",
      foregroundMode: "light",
    },
    supportsMultipleConnections: true,
  },
};
