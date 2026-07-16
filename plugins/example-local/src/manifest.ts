import {
  RAYVAN_PLUGIN_API_VERSION,
  type PluginManifest,
} from "@rayvan/plugin-sdk";

export const EXAMPLE_LOCAL_PLUGIN_ID = "example-local";
export const LOCAL_SERVICE_RESOURCE_TYPE = "local.service";
export const LOCAL_SERVICE_SCHEMA_VERSION = "1.0.0";

export const manifest: PluginManifest = {
  id: EXAMPLE_LOCAL_PLUGIN_ID,
  name: "Example Local",
  description:
    "Built-in mock local environment plugin that demonstrates the Rayvan plugin lifecycle without external credentials.",
  version: "0.1.0",
  publisher: "rayvan",
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [
    "discover",
    "inspect",
    "plan",
    "apply",
    "verify",
    "evaluate_findings",
  ],
  permissions: [],
  resourceTypes: [
    {
      id: LOCAL_SERVICE_RESOURCE_TYPE,
      name: "Local Service",
      description: "An in-memory mock local development service",
      schemaVersion: LOCAL_SERVICE_SCHEMA_VERSION,
    },
  ],
  findingRules: [
    {
      id: `${EXAMPLE_LOCAL_PLUGIN_ID}.service-present`,
      name: "Local service present",
      description:
        "Emits an informational finding when at least one local service is discoverable.",
      category: "resource",
      defaultSeverity: "info",
    },
  ],
  presentation: {
    icon: {
      iconId: "example-local",
      initials: "EL",
      label: "Example Local",
    },
    theme: {
      surface: "neutral",
      accentColor: "#64748B",
      foregroundMode: "dark",
    },
    supportsMultipleConnections: false,
  },
};
