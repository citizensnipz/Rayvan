import {
  RAYVAN_PLUGIN_API_VERSION,
  type PluginManifest,
} from "@rayvan/plugin-sdk";

import {
  GITHUB_ACTIONS_VARIABLES_RESOURCE_TYPE,
  GITHUB_ACTIONS_VARIABLES_SCHEMA_VERSION,
  GITHUB_PLUGIN_BINARY,
  GITHUB_PLUGIN_ID,
  GITHUB_REPOSITORY_RESOURCE_TYPE,
  GITHUB_REPOSITORY_SCHEMA_VERSION,
} from "./constants.js";

export const manifest: PluginManifest = {
  id: GITHUB_PLUGIN_ID,
  name: "GitHub",
  description:
    "Connect Rayvan to GitHub repositories and manage Actions variables.",
  version: "0.1.0",
  publisher: "rayvan",
  publisherInfo: {
    name: "Rayvan",
    url: "https://rayvan.dev",
    supportUrl: "https://github.com/rayvan/rayvan",
  },
  rayvanApiVersion: RAYVAN_PLUGIN_API_VERSION,
  capabilities: [
    "authenticate",
    "discover",
    "inspect",
    "plan",
    "apply",
    "verify",
    "evaluate_findings",
  ],
  permissions: ["network", "read_secrets", "write_remote_configuration"],
  resourceTypes: [
    {
      id: GITHUB_REPOSITORY_RESOURCE_TYPE,
      name: "GitHub Repository",
      description: "A GitHub repository visible to the authenticated account",
      schemaVersion: GITHUB_REPOSITORY_SCHEMA_VERSION,
    },
    {
      id: GITHUB_ACTIONS_VARIABLES_RESOURCE_TYPE,
      name: "Actions Repository Variables",
      description:
        "GitHub Actions variables for a repository (readable values; secrets are presence-only)",
      schemaVersion: GITHUB_ACTIONS_VARIABLES_SCHEMA_VERSION,
    },
  ],
  findingRules: [
    {
      id: `${GITHUB_PLUGIN_ID}.unused-actions-variable`,
      name: "Unused Actions variable",
      description:
        "An Actions repository variable exists but is not referenced by discovered workflow files.",
      category: "configuration",
      defaultSeverity: "warning",
    },
    {
      id: `${GITHUB_PLUGIN_ID}.missing-referenced-variable`,
      name: "Missing referenced variable",
      description:
        "A workflow references vars.NAME but no matching Actions repository variable was found.",
      category: "configuration",
      defaultSeverity: "error",
    },
  ],
  entrypoints: [
    {
      runtime: "native",
      binary: GITHUB_PLUGIN_BINARY,
      protocolVersion: "1",
    },
  ],
  networkHosts: ["api.github.com", "github.com"],
  setup: {
    authMethods: ["github_device_flow", "pat"],
    steps: [
      {
        id: "choose-auth",
        title: "Choose authentication",
        description: "Use GitHub device flow or a personal access token.",
        widget: "auth_method_select",
      },
      {
        id: "device-flow",
        title: "Authorize with GitHub",
        description: "Enter the code shown in your browser.",
        widget: "device_flow",
      },
      {
        id: "pat-input",
        title: "Personal access token",
        description: "Paste a classic or fine-grained PAT with Actions variables scope.",
        widget: "pat_input",
      },
      {
        id: "confirm",
        title: "Confirm connection",
        widget: "confirm",
      },
    ],
  },
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
